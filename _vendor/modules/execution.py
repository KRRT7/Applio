"""ONNX Runtime execution-provider helpers shared by the CLI and the library."""

from __future__ import annotations

import os
from typing import List

import onnxruntime


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [p.replace("ExecutionProvider", "").lower() for p in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available = onnxruntime.get_available_providers()
    return [
        provider
        for provider, encoded in zip(available, encode_execution_providers(available))
        if any(wanted in encoded for wanted in execution_providers)
    ]


def suggest_default_execution_provider() -> str:
    """Pick the best available provider: cuda > rocm > coreml > openvino > dml > cpu."""
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ("cuda", "rocm", "coreml", "openvino", "dml"):
        if pref in available:
            return pref
    return "cpu"


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads(execution_providers: List[str]) -> int:
    """Suggest a thread count for the given decoded providers."""
    for single in ("DmlExecutionProvider", "ROCMExecutionProvider", "OpenVINOExecutionProvider"):
        if single in execution_providers:
            return 1
    if "CUDAExecutionProvider" in execution_providers:
        return 2
    cpu_count = os.cpu_count() or 4
    return max(4, min(cpu_count - 2, 16))
