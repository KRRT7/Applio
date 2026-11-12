import json
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
elif sys.platform == "win32":
    # Ensure virtualenv Scripts and torch DLL directories are accessible
    # for dynamic loading of ROCm and runtime libraries
    scripts_dir = os.path.dirname(sys.executable)
    if os.path.isdir(scripts_dir):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(scripts_dir)
            except Exception:
                pass
        curr_path = os.environ.get("PATH", "")
        if scripts_dir.lower() not in curr_path.lower():
            os.environ["PATH"] = f"{scripts_dir};{curr_path}"

    venv_base = os.path.dirname(scripts_dir)
    torch_lib = os.path.join(venv_base, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
        curr_path = os.environ.get("PATH", "")
        if torch_lib.lower() not in curr_path.lower():
            os.environ["PATH"] = f"{torch_lib};{curr_path}"

    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ.setdefault("MIOPEN_DEBUG_DISABLE_FIND_DB", "1")
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
    os.environ.setdefault("AMD_COMGR_CACHE", "0")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")
from functools import lru_cache

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

current_script_directory = os.path.dirname(os.path.realpath(__file__))
if current_script_directory not in sys.path:
    sys.path.append(current_script_directory)


@lru_cache(maxsize=None)
def import_voice_converter():
    try:
        from rvc.infer.infer import VoiceConverter

        return VoiceConverter()
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. Please install requirements via 'uv sync'"
        )


def _base_infer_kwargs(
    pitch=0,
    index_rate=0.3,
    volume_envelope=1.0,
    protect=0.33,
    f0_method="rmvpe",
    pth_path="",
    index_path="",
    f0_autotune=False,
    f0_autotune_strength=1.0,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
    export_format="WAV",
    embedder_model="contentvec",
    embedder_model_custom=None,
    formant_shifting=False,
    formant_qfrency=1.0,
    formant_timbre=1.0,
    sid=0,
    inference_engine=None,
    onnx_precision="auto",
    cleanup_mode="off",
    cleanup_strength=12,
):
    return {
        "model_path": pth_path,
        "index_path": index_path,
        "volume_envelope": volume_envelope,
        "pitch": pitch,
        "index_rate": index_rate,
        "protect": protect,
        "f0_method": f0_method,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "proposed_pitch": proposed_pitch,
        "proposed_pitch_threshold": proposed_pitch_threshold,
        "export_format": export_format,
        "embedder_model": embedder_model,
        "embedder_model_custom": embedder_model_custom,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "sid": sid,
        "inference_engine": inference_engine,
        "onnx_precision": onnx_precision,
        "cleanup_mode": cleanup_mode,
        "cleanup_strength": cleanup_strength,
    }


def run_infer_script(
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    export_format: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    sid: int = 0,
    inference_engine: str = None,
    onnx_precision: str = "auto",
    cleanup_mode: str = "off",
    cleanup_strength: float = 12,
):
    kwargs = _base_infer_kwargs(
        pitch=pitch,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        f0_method=f0_method,
        pth_path=pth_path,
        index_path=index_path,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        formant_shifting=formant_shifting,
        formant_qfrency=formant_qfrency,
        formant_timbre=formant_timbre,
        sid=sid,
        inference_engine=inference_engine,
        onnx_precision=onnx_precision,
        cleanup_mode=cleanup_mode,
        cleanup_strength=cleanup_strength,
    )
    kwargs["audio_input_path"] = input_path
    kwargs["audio_output_path"] = output_path
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(**kwargs)
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )


def run_batch_infer_script(
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_path: str,
    index_path: str,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    export_format: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    sid: int = 0,
    inference_engine: str = None,
    onnx_precision: str = "auto",
    cleanup_mode: str = "off",
    cleanup_strength: float = 12,
):
    kwargs = _base_infer_kwargs(
        pitch=pitch,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        f0_method=f0_method,
        pth_path=pth_path,
        index_path=index_path,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        formant_shifting=formant_shifting,
        formant_qfrency=formant_qfrency,
        formant_timbre=formant_timbre,
        sid=sid,
        inference_engine=inference_engine,
        onnx_precision=onnx_precision,
        cleanup_mode=cleanup_mode,
        cleanup_strength=cleanup_strength,
    )
    kwargs["audio_input_paths"] = input_folder
    kwargs["audio_output_path"] = output_folder
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio_batch(**kwargs)
    return f"Files from {input_folder} inferred successfully."


def run_prerequisites_script(models: bool = True, exe: bool = True):
    try:
        from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

        prequisites_download_pipeline(models, exe)
        return "Prerequisites downloaded and installed successfully."
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. Please install requirements: uv sync"
        )


import click


def _get_version():
    config_path = os.path.join(
        current_script_directory, "assets", "config_template.json"
    )
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


VERSION = _get_version()


def _infer_opts(func):
    """Core inference options shared by infer and batch_infer."""
    opts = [
        click.option(
            "--cleanup-mode",
            type=click.Choice(["off", "light", "speech"]),
            default="off",
            help="Optional file noise reduction. Speech may remove singing.",
        ),
        click.option(
            "--cleanup-strength",
            type=click.FloatRange(1, 30),
            default=12,
            help="Noise reduction limit in dB.",
        ),
        click.option(
            "--pitch",
            type=click.IntRange(-24, 24),
            default=0,
            help="Set the pitch of the audio. Higher values result in a higher pitch.",
        ),
        click.option(
            "--index-rate",
            type=click.FloatRange(0, 1),
            default=0.3,
            help="Control the influence of the index file on the output.",
        ),
        click.option(
            "--volume-envelope",
            type=click.FloatRange(0, 1),
            default=1.0,
            help="Control the blending of the output's volume envelope.",
        ),
        click.option(
            "--protect",
            type=click.FloatRange(0, 0.5),
            default=0.33,
            help="Protect consonants and breathing sounds from artifacts.",
        ),
        click.option(
            "--f0-method",
            type=click.Choice(
                [
                    "crepe",
                    "rmvpe",
                    "fcpe",
                    "hybrid[crepe+rmvpe]",
                    "hybrid[crepe+fcpe]",
                    "hybrid[rmvpe+fcpe]",
                    "hybrid[crepe+rmvpe+fcpe]",
                ]
            ),
            default="rmvpe",
            help="Choose the pitch extraction algorithm.",
        ),
        click.option(
            "--f0-autotune",
            is_flag=True,
            default=False,
            help="Apply a light autotune to the inferred audio.",
        ),
        click.option(
            "--f0-autotune-strength",
            type=click.FloatRange(0, 1),
            default=1.0,
            help="Autotune strength (higher = more chromatic snap).",
        ),
        click.option(
            "--proposed-pitch",
            is_flag=True,
            default=False,
            help="Enable proposed pitch adjustment.",
        ),
        click.option(
            "--proposed-pitch-threshold",
            type=click.FloatRange(50, 1199),
            default=155.0,
            help="Proposed pitch threshold value.",
        ),
        click.option(
            "--export-format",
            type=click.Choice(["WAV", "MP3", "FLAC", "OGG", "M4A"]),
            default="WAV",
            help="Output audio format.",
        ),
        click.option(
            "--embedder-model",
            type=click.Choice(
                [
                    "contentvec",
                    "spin",
                    "spin-v2",
                    "chinese-hubert-base",
                    "japanese-hubert-base",
                    "korean-hubert-base",
                    "custom",
                ]
            ),
            default="contentvec",
            help="Model used for generating speaker embeddings.",
        ),
        click.option(
            "--embedder-model-custom",
            type=str,
            default=None,
            help="Path to a custom embedding model (only when --embedder-model is 'custom').",
        ),
        click.option(
            "--sid", type=int, default=0, help="Speaker ID for multi-speaker models."
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


def _formant_opts(func):
    """Formant options shared by infer and batch_infer."""
    opts = [
        click.option(
            "--formant-shifting",
            is_flag=True,
            default=False,
            help="Apply formant shifting to the input audio.",
        ),
        click.option(
            "--formant-qfrency",
            type=float,
            default=1.0,
            help="Formant shift frequency.",
        ),
        click.option(
            "--formant-timbre", type=float, default=1.0, help="Formant shift timbre."
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


@click.group(invoke_without_command=True)
@click.version_option(
    version=VERSION, prog_name="Applio", message="%(prog)s v%(version)s"
)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option("--input-path", required=True, help="Full path to the input audio file.")
@click.option(
    "--output-path", required=True, help="Full path to the output audio file."
)
@click.option(
    "--pth-path", required=True, help="Full path to the RVC model file (.pth)."
)
@click.option(
    "--index-path", required=True, help="Full path to the index file (.index)."
)
@_infer_opts
@_formant_opts
def infer(**kwargs):
    """Run voice conversion on a single audio file."""
    result = run_infer_script(**kwargs)
    click.echo(result[0])


@cli.command()
@click.option(
    "--input-folder", required=True, help="Folder containing input audio files."
)
@click.option(
    "--output-folder", required=True, help="Folder for saving output audio files."
)
@click.option(
    "--pth-path", required=True, help="Full path to the RVC model file (.pth)."
)
@click.option(
    "--index-path", required=True, help="Full path to the index file (.index)."
)
@_infer_opts
@_formant_opts
def batch_infer(**kwargs):
    """Run voice conversion on multiple audio files in a folder."""
    result = run_batch_infer_script(**kwargs)
    click.echo(result)


@cli.command()
@click.option("--models/--no-models", default=True, help="Download additional models.")
@click.option("--exe/--no-exe", default=True, help="Download required executables.")
def prerequisites(**kwargs):
    """Install prerequisites for RVC."""
    result = run_prerequisites_script(kwargs["models"], kwargs["exe"])
    click.echo(result)


def main():
    cli()


if __name__ == "__main__":
    main()
