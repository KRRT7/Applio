"""Tests for scripts/vendor_dlc.py, which generates _vendor/ from upstream
Deep-Live-Cam plus patches. Whether the committed trees are current is CI's
`vendor_dlc.py check` (it needs the network); these cover the pure parts."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vendor_dlc", ROOT / "scripts" / "vendor_dlc.py")
vendor_dlc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor_dlc)
rewrite = vendor_dlc.rewrite_imports

NOTE = "  # noqa: F401  (loads the submodule; vendoring-safe form of `import modules.{}`)"


def test_from_imports_move_under_vendor():
    assert rewrite("from modules import imread_unicode\n") == "from _vendor.modules import imread_unicode\n"
    assert rewrite("from modules.core import update_status\n") == "from _vendor.modules.core import update_status\n"
    assert rewrite("    from modules.processors.frame.core import x\n") == (
        "    from _vendor.modules.processors.frame.core import x\n")


def test_import_modules_binds_the_same_name():
    assert rewrite("import modules\n") == "from _vendor import modules\n"


def test_submodule_import_binds_modules_once_and_loads_the_submodule():
    src = "import cv2\nimport modules.globals  # a comment\nimport modules.processors.frame.core\n"
    assert rewrite(src) == (
        "import cv2\n"
        "from _vendor import modules  # noqa: F401\n"
        "import _vendor.modules.globals as _modules_globals" + NOTE.format("globals") + "\n"
        "import _vendor.modules.processors.frame.core as _modules_frame_core"
        + NOTE.format("processors.frame.core") + "\n"
    )


def test_namespace_import_starts_its_own_group_after_a_blank_line():
    out = rewrite("import threading\n\nimport modules.globals\n")
    assert out.splitlines()[2:4] == ["from _vendor import modules  # noqa: F401", ""]


def test_no_extra_binding_when_modules_is_already_imported():
    out = rewrite("import modules\nimport modules.globals\n")
    assert out.count("from _vendor import modules") == 1


def test_leaves_everything_else_alone():
    src = (
        "x = modules.globals.det_size\n"
        "import modules_extra\n"
        "importlib.import_module(f'modules.processors.frame.{name}')\n"
        "# from modules import nothing\n"
    )
    assert rewrite(src) == src


def test_line_endings_are_kept():
    assert rewrite("from modules import a\r\n") == "from _vendor.modules import a\r\n"


def test_check_ignores_crlf_from_a_windows_checkout(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert vendor_dlc._same(a, b)
    b.write_bytes(b"x = 1\r\ny = 3\r\n")
    assert not vendor_dlc._same(a, b)


def test_patch_series_is_applied_in_name_order(tmp_path):
    for name in ("0010-b.patch", "0002-a.patch", "notes.txt", "0001-c.patch"):
        (tmp_path / name).write_text("")
    assert [p.name for p in vendor_dlc.find_patches(tmp_path)] == ["0001-c.patch", "0002-a.patch", "0010-b.patch"]


def test_the_committed_series_and_pin_exist():
    pin = vendor_dlc.UPSTREAM_FILE.read_text(encoding="utf-8").strip()
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
    assert vendor_dlc.find_patches(), "no patches in tools/vendoring/patches/dlc"
    for patch in vendor_dlc.find_patches():
        assert b"\r\n" not in patch.read_bytes(), f"{patch.name} has CRLF; see .gitattributes"


@pytest.mark.skipif(not os.environ.get("APPLIO_NETWORK_TESTS"), reason="needs the network: set APPLIO_NETWORK_TESTS=1")
def test_committed_trees_are_current():
    assert vendor_dlc.check() == 0
