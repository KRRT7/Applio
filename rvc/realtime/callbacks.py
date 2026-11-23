import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from rvc.realtime.audio import Audio
from rvc.realtime.constants import AUDIO_SAMPLE_RATE
from rvc.realtime.worker import VoiceChangerWorker

# Fade applied where converted audio stops or starts mid-stream (a late block,
# a dropped result): long enough to remove the click, too short to hear.
EDGE_FADE_S = 0.005


class AudioCallbacks:
    def __init__(
        self,
        pass_through: bool = False,
        block_frame: int = 24576,
        cross_fade_overlap_size: float = 0.1,
        extra_convert_size: float = 0.5,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = -90,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        formant_shifting: bool = False,
        formant_qfrency: float = 1.0,
        formant_timbre: float = 1.0,
        record_audio: bool = False,
        record_audio_path: str = None,
        export_format: str = "WAV",
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        **kwargs,
        # device: str = "cuda",
    ):
        self.pass_through = pass_through
        self._last_output = None
        self._last_vol = 0
        # Audio-path safety: a block the worker did not deliver in time is
        # replaced by a short fade to silence (replaying the previous block
        # made the listener hear the same syllable twice), and audio resumes
        # with a short fade-in, so neither edge clicks.
        self._fade = max(1, int(audio_sample_rate * EDGE_FADE_S))
        self._in_gap = False
        self.late_blocks = 0  # blocks the worker did not deliver in time
        self.voice_failed = False  # the worker process died; output is silence

        vc_kwargs = dict(
            block_frame=block_frame,
            cross_fade_overlap_size=cross_fade_overlap_size,
            extra_convert_size=extra_convert_size,
            model_path=model_path,
            index_path=index_path,
            f0_method=f0_method,
            embedder_model=embedder_model,
            embedder_model_custom=embedder_model_custom,
            silent_threshold=silent_threshold,
            vad_enabled=vad_enabled,
            vad_sensitivity=vad_sensitivity,
            vad_frame_ms=vad_frame_ms,
            sid=sid,
            formant_shifting=formant_shifting,
            formant_qfrency=formant_qfrency,
            formant_timbre=formant_timbre,
            record_audio=record_audio,
            record_audio_path=record_audio_path,
            export_format=export_format,
            audio_sample_rate=audio_sample_rate,
            index_rate=index_rate,
            **kwargs,
        )
        self.vc = VoiceChangerWorker(vc_kwargs)
        self.vc.start()

        self.audio = Audio(
            self,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
            input_audio_gain,
            output_audio_gain,
            monitor_audio_gain,
            monitor,
        )

    def change_voice(
        self,
        received_data: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.pass_through:
            vol = float(np.sqrt(np.square(received_data).mean(dtype=np.float32)))
            return received_data, vol, 0

        params = dict(
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            protect=protect,
            volume_envelope=volume_envelope,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            proposed_pitch=proposed_pitch,
            proposed_pitch_threshold=proposed_pitch_threshold,
        )
        n = received_data.shape[0]
        if self.voice_failed or self._worker_died():
            # Never stall or loop the last block: a dead worker means silence
            # (the UI reports the failure from voice_failed).
            self.voice_failed = True
            return self._gap(n), 0, 0

        self.vc.submit(received_data, params)

        result = self.vc.retrieve()
        if result is not None:
            audio, vol, perf_ms, _warmup = result
            skipped = getattr(self.vc, "last_skipped", 0)
            if (self._in_gap or skipped) and audio.shape[0] == n:
                audio = audio.copy()
                k = min(self._fade, n)
                audio[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
            self._in_gap = False
            self._last_output = audio
            self._last_vol = vol
            return audio, vol, perf_ms

        # Not ready in time (or not loaded yet).
        if self._last_output is not None:
            self.late_blocks += 1
        return self._gap(n), 0, 0

    def _worker_died(self) -> bool:
        process = getattr(self.vc, "_process", None)
        return process is not None and not process.is_alive()

    def _gap(self, n: int) -> np.ndarray:
        """Silence, fading out from the last sample played if audio was flowing."""
        out = np.zeros(n, dtype=np.float32)
        last = self._last_output
        if not self._in_gap and last is not None and len(last):
            k = min(self._fade, n)
            out[:k] = float(last[-1]) * np.linspace(
                1.0, 0.0, k, endpoint=False, dtype=np.float32
            )
        self._in_gap = True
        self._last_vol = 0
        return out
