"""The UI process stays light: no torch, and no UI rebuilt in spawned children.

Torch only runs in the voice worker. In the UI process it cost ~0.3 GB of
RAM (its CUDA runtime) on top of ONNX Runtime's, which is all face swap
needs. Each check runs in a fresh interpreter, since this test process may
already have torch loaded.
"""

import os
import pickle
import subprocess
import sys
import textwrap
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
    )


def test_ui_modules_do_not_import_torch():
    # What app.build_app() and the Live page load, plus the face pipeline's
    # processors and the parent side of the voice worker.
    result = _run(
        """
        import sys
        sys.argv = ["app.py"]
        import tabs.settings.settings, tabs.faceswap.faceswap, tabs.realtime.realtime
        import rvc.realtime.callbacks, rvc.realtime.worker
        from _vendor import deep_live_cam
        from _vendor.modules.processors.frame import face_swapper, face_enhancer, face_parser
        from _vendor.modules import face_analyser, platform_info
        print("TORCH", "torch" in sys.modules)
        """
    )
    assert "TORCH False" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]


def test_spawned_children_do_not_rebuild_the_ui():
    # multiprocessing "spawn" re-runs the main script as __mp_main__ in every
    # child (the voice worker); app.py must not build the UI then.
    result = _run(
        """
        import runpy, sys
        sys.argv = ["app.py"]
        runpy.run_path("app.py", run_name="__mp_main__")
        print("HEAVY", any(m in sys.modules for m in ("gradio", "torch", "insightface")))
        """
    )
    assert "HEAVY False" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]


def test_checkpoint_metadata_matches_torch_load(tmp_path):
    torch = pytest.importorskip("torch")
    from rvc.lib.checkpoint_meta import read_checkpoint_metadata

    path = tmp_path / "voice.pth"
    torch.save(
        {"weight": {"emb_g.weight": torch.zeros(3, 4)}, "speakers_id": 3, "config": [1, 2, 48000], "f0": 1},
        path,
    )
    meta = read_checkpoint_metadata(str(path))
    assert meta["speakers_id"] == 3
    assert meta["config"] == [1, 2, 48000]
    assert meta["weight"]["emb_g.weight"] is None  # tensors are never read


def test_checkpoint_metadata_refuses_arbitrary_objects(tmp_path):
    from rvc.lib.checkpoint_meta import read_checkpoint_metadata

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    path = tmp_path / "evil.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("evil/data.pkl", pickle.dumps({"speakers_id": Evil()}))
    with pytest.raises(pickle.UnpicklingError):
        read_checkpoint_metadata(str(path))


def test_checkpoint_metadata_rejects_non_zip(tmp_path):
    from rvc.lib.checkpoint_meta import read_checkpoint_metadata

    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        read_checkpoint_metadata(str(path))
