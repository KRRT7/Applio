import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import traceback

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

# Keep a reference to real standard streams before redirection
real_stdout = sys.stdout
real_stderr = sys.stderr

active_job_id = None


def get_active_job_id():
    return active_job_id


class IPCStdoutRedirector:
    def __init__(self, target, job_id_fn):
        self.target = target
        self.job_id_fn = job_id_fn
        self.buffer = ""

    def write(self, s):
        if not s:
            return
        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                send_ipc(
                    {
                        "type": "log",
                        "id": self.job_id_fn(),
                        "message": line,
                    }
                )

    def flush(self):
        if self.buffer.strip():
            send_ipc(
                {
                    "type": "log",
                    "id": self.job_id_fn(),
                    "message": self.buffer.strip(),
                }
            )
        self.buffer = ""


def send_ipc(data: dict):
    payload = {"_applio_ipc": True, **data}
    real_stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    real_stdout.flush()


def to_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)


def map_params(params: dict, input_path: str, output_path: str) -> dict:
    return {
        "pitch": int(params.get("pitch", 0)),
        "index_rate": float(params.get("indexRate", 0.75)),
        "volume_envelope": float(params.get("volumeEnvelope", 1.0)),
        "protect": float(params.get("protect", 0.5)),
        "f0_method": str(params.get("f0Method", "rmvpe")),
        "inference_engine": params.get("inferenceEngine", "torch"),
        "onnx_precision": params.get("onnxPrecision", "auto"),
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": params.get("pthPath", ""),
        "index_path": params.get("indexPath", "") or "",
        "f0_autotune": to_bool(params.get("f0Autotune", False)),
        "f0_autotune_strength": float(params.get("f0AutotuneStrength", 1.0)),
        "proposed_pitch": to_bool(params.get("proposedPitch", False)),
        "proposed_pitch_threshold": float(params.get("proposedPitchThreshold", 155.0)),
        "export_format": str(params.get("exportFormat", "WAV")),
        "embedder_model": str(params.get("embedderModel", "contentvec")),
        "embedder_model_custom": params.get("embedderModelCustom") or None,
        "formant_shifting": to_bool(params.get("formantShifting", False)),
        "formant_qfrency": float(params.get("formantQfrency", 1.0)),
        "formant_timbre": float(params.get("formantTimbre", 1.0)),
        "sid": int(params.get("sid", 0)),
        "cleanup_mode": params.get("cleanupMode", "off"),
        "cleanup_strength": float(params.get("cleanupStrength", 12)),
    }


def run_worker():
    global active_job_id

    # Install redirectors
    sys.stdout = IPCStdoutRedirector(real_stdout, get_active_job_id)
    sys.stderr = IPCStdoutRedirector(real_stdout, get_active_job_id)

    from rvc.infer.infer import VoiceConverter

    # Preload singleton VoiceConverter in background
    vc = VoiceConverter()

    send_ipc({"type": "ready", "pid": os.getpid()})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            send_ipc({"type": "error", "id": None, "error": f"Invalid JSON: {e}"})
            continue

        cmd = req.get("command")
        job_id = req.get("id")
        active_job_id = job_id

        try:
            if cmd == "ping":
                send_ipc({"type": "pong", "id": job_id})
            elif cmd == "warmup":
                # Warm up CUDA, PyTorch, audio libraries, embedder, and the most recent model
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.init()

                    # Pre-import heavy numerical and DSP libraries
                    import librosa
                    import soundfile

                    vc.load_hubert("contentvec", None)

                    # Look for the latest model in logs/ to preload so the first user conversion is instant
                    preloaded_name = None
                    from rvc.lib.user_config import get_logs_dir

                    logs_dir = get_logs_dir()
                    if os.path.isdir(logs_dir):
                        candidates = []
                        for r, _, files in os.walk(logs_dir):
                            for f in files:
                                if f.lower().endswith(".pth") and not f.startswith(
                                    ("G_", "D_")
                                ):
                                    full_p = os.path.join(r, f)
                                    try:
                                        candidates.append(
                                            (os.path.getmtime(full_p), full_p)
                                        )
                                    except Exception:
                                        pass
                        if candidates:
                            candidates.sort(reverse=True)
                            latest_model = candidates[0][1]
                            try:
                                vc.get_vc(latest_model, 0)
                                preloaded_name = os.path.basename(latest_model)
                            except Exception:
                                pass

                    msg = f"Worker warmed up (embedder: contentvec" + (
                        f", active model: {preloaded_name})" if preloaded_name else ")"
                    )
                    send_ipc({"type": "log", "id": job_id, "message": msg})
                except Exception as e:
                    send_ipc(
                        {"type": "log", "id": job_id, "message": f"Warmup notice: {e}"}
                    )
            elif cmd == "unload":
                vc.cleanup_model()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                send_ipc({"type": "unloaded", "id": job_id})
            elif cmd == "infer":
                params = req.get("params", {})
                input_path = req.get("inputPath")
                output_path = req.get("outputPath")
                kwargs = map_params(params, input_path, output_path)

                export_format = kwargs.get("export_format", "WAV")
                vc.convert_audio(**kwargs)
                final_out = (
                    os.path.splitext(output_path)[0] + f".{export_format.lower()}"
                )
                info_msg = f"File {input_path} inferred successfully."
                sys.stdout.flush()
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "outputPath": final_out,
                        "info": info_msg,
                    }
                )
            elif cmd == "infer_batch":
                params = req.get("params", {})
                input_folder = req.get("inputFolder")
                output_folder = req.get("outputFolder")
                kwargs = map_params(params, "", "")
                kwargs.pop("audio_input_path")
                kwargs.pop("audio_output_path")

                vc.convert_audio_batch(
                    audio_input_paths=input_folder,
                    audio_output_path=output_folder,
                    **kwargs,
                )
                sys.stdout.flush()
                send_ipc(
                    {
                        "type": "done",
                        "id": job_id,
                        "success": True,
                        "info": f"Batch conversion in {input_folder} completed.",
                    }
                )
            elif cmd == "preload_model":
                pth_path = req.get("pthPath")
                sid = int(req.get("sid", 0))
                if pth_path and os.path.isfile(pth_path):
                    vc.get_vc(pth_path, sid)
                    send_ipc(
                        {
                            "type": "done",
                            "id": job_id,
                            "success": True,
                            "info": f"Model '{os.path.basename(pth_path)}' preloaded.",
                        }
                    )
                else:
                    send_ipc(
                        {
                            "type": "error",
                            "id": job_id,
                            "error": f"Model file not found: {pth_path}",
                        }
                    )
            else:
                send_ipc(
                    {
                        "type": "error",
                        "id": job_id,
                        "error": f"Unknown command: {cmd}",
                    }
                )
        except Exception as e:
            sys.stdout.flush()
            tb = traceback.format_exc()
            send_ipc(
                {
                    "type": "error",
                    "id": job_id,
                    "error": str(e),
                    "traceback": tb,
                }
            )
        finally:
            active_job_id = None


if __name__ == "__main__":
    run_worker()
