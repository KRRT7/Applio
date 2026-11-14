"""ONNX Runtime's CUDA provider loads with torch's CUDA/cuDNN DLLs alone.

Windows has no nvidia-* wheels: onnxruntime-gpu uses the DLLs the cu128
torch wheel ships in torch/lib. Each entry point must make them findable
before an ONNX session is created, so check them in a fresh process.
"""

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.skipif(
    sys.platform != "win32"
    or not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="Windows + CUDA + onnxruntime-gpu only",
)

# Build a one-node graph in-process, create a CUDA session, report providers.
_SESSION = textwrap.dedent("""
    import numpy as np, onnx, onnxruntime as ort
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    w = helper.make_tensor("w", TensorProto.FLOAT, [3, 1, 3, 3], [1.0] * 27)
    node = helper.make_node("Conv", ["x", "w"], ["y"], group=3, pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "g", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    s = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    s.run(None, {"x": np.ones((1, 3, 8, 8), np.float32)})
    print("PROVIDERS", s.get_providers()[0])
    """)


@pytest.mark.parametrize(
    "setup",
    [
        "import core",  # app.py / the voice worker's parent
        "import torch",  # voice worker: rvc.realtime.core imports torch first
        "from _vendor import deep_live_cam",  # face swap without core
        "import onnxruntime as ort; ort.preload_dlls()",  # wrapper.py / onnx_decoder.py
    ],
)
def test_cuda_provider_loads_from_torch_dlls(setup):
    result = subprocess.run(
        [sys.executable, "-c", setup + "\n" + _SESSION],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
    )
    assert "PROVIDERS CUDAExecutionProvider" in result.stdout, result.stderr[-2000:]
