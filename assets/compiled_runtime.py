"""Filesystem layout for the Nuitka-compiled build.

Applio resolves its files relative to the working directory. A onefile build
unpacks to a versioned cache folder (see build-macos.sh), so on start-up the
program changes into that folder and links the user's data (settings, voice
models, recordings) to a persistent per-user folder. The data then survives
upgrades, which unpack each version into a new cache folder. Voice models
shipped with the build stay in the program folder and are used read-only.

Running from source never imports this module.
"""

import os
import shutil
from pathlib import Path

from rvc.lib.user_config import config_path

# Paths inside the unpacked program -> names inside the per-user data folder.
USER_DATA_LINKS = {
    "assets/config.json": "config.json",
    "logs": "logs",
    "assets/audios": "audios",
}


def user_data_dir():
    """Persistent per-user folder (~/Library/Application Support/Applio on macOS)."""
    override = os.environ.get("APPLIO_DATA_DIR")
    return Path(override).expanduser() if override else config_path().parent


def _link(source, target):
    if source.is_symlink():
        if Path(os.readlink(source)) == target:
            return
        source.unlink()
    elif source.is_dir():
        shutil.rmtree(source)
    elif source.exists():
        source.unlink()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target, target_is_directory=target.is_dir())


def prepare(resource_dir):
    """Change into the unpacked program folder and link user data into it."""
    resource_dir = Path(resource_dir)
    data_dir = user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(resource_dir)

    for relative, name in USER_DATA_LINKS.items():
        target = data_dir / name
        if name != "config.json":
            # app.py creates config.json from the template when it is missing.
            target.mkdir(parents=True, exist_ok=True)
        _link(resource_dir / relative, target)

    # Voice models shipped with the build are read in place (read-only), listed
    # next to the user's own; see rvc.lib.user_config.BUNDLED_LOGS_PREFIX.
    bundled = resource_dir / "bundled" / "logs"
    if bundled.is_dir():
        os.environ["APPLIO_BUNDLED_LOGS_DIR"] = str(bundled)

    # Realtime resolves model paths against the logs folder; keep it in the
    # persistent location for spawned worker processes too.
    os.environ.setdefault("APPLIO_LOGS_DIR", str(data_dir / "logs"))
    return data_dir
