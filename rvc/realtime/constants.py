"""Realtime audio constants.

Kept free of heavy imports: the UI process (audio devices, settings, the
worker's parent side) needs these numbers without loading torch, which only
the voice worker uses.
"""

SAMPLE_RATE = 16000  # model-side rate
AUDIO_SAMPLE_RATE = 48000  # device-side rate
