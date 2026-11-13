"""Live profiles: everything that belongs to one person, saved together.

A profile holds a person's face photos (copied into the profile so it does
not depend on files elsewhere), the voice model and index to use, the voice
parameters tuned for that voice, and the face "look" settings tuned for
that face. Machine and call settings (camera, resolution, audio devices,
virtual camera, video delay, performance options) are not part of a
profile.

Layout (``logs/`` is git-ignored, like Applio's voice models)::

    logs/profiles/<slug>/profile.json
    logs/profiles/<slug>/faces/01_<photo>.jpg ...

No UI code here, so it can also be used from scripts.
"""

import json
import os
import re
import shutil
import time

PROFILES_DIR = os.path.join(os.getcwd(), "logs", "profiles")

FACE_KEYS = ("swap_model", "enhancer", "mouth_mask", "sharpness", "color_correction", "pixel_boost")
VOICE_KEYS = ("model_file", "index_file", "pitch", "index_rate", "protect", "f0_method")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def slug(name):
    """Folder name for a profile name ("Ana María" -> "ana_maría")."""
    s = re.sub(r"[^\w\- ]", "", str(name or ""), flags=re.UNICODE).strip().lower()
    return re.sub(r"\s+", "_", s)


def _dir(name):
    s = slug(name)
    if not s:
        raise ValueError("Profile name is empty.")
    return os.path.join(PROFILES_DIR, s)


def _portable(path):
    """Store paths with forward slashes, relative to Applio when possible."""
    if not path:
        return ""
    p = os.path.relpath(path, os.getcwd()) if os.path.isabs(path) else path
    return p.replace("\\", "/")


def list_profiles():
    """Display names of saved profiles, sorted."""
    names = []
    if os.path.isdir(PROFILES_DIR):
        for entry in os.listdir(PROFILES_DIR):
            path = os.path.join(PROFILES_DIR, entry, "profile.json")
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        names.append(json.load(f).get("name") or entry)
                except (OSError, ValueError):
                    names.append(entry)
    return sorted(names, key=str.lower)


def load_profile(name):
    """The profile as a dict with 'name', 'face', 'voice' and absolute 'faces'."""
    d = _dir(name)
    with open(os.path.join(d, "profile.json"), encoding="utf-8") as f:
        data = json.load(f)
    faces_dir = os.path.join(d, "faces")
    data["faces"] = sorted(
        os.path.join(faces_dir, f) for f in os.listdir(faces_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ) if os.path.isdir(faces_dir) else []
    data.setdefault("face", {})
    data.setdefault("voice", {})
    return data


def save_profile(name, faces, face_settings, voice_settings):
    """Create or update a profile. Photos are copied into the profile's
    faces/ folder (in the given order); photos removed from the list are
    deleted from it. Returns the saved profile (as load_profile would)."""
    d = _dir(name)
    faces_dir = os.path.join(d, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    faces_dir_abs = os.path.abspath(faces_dir)

    tmp_dir = os.path.join(d, ".incoming")
    os.makedirs(tmp_dir, exist_ok=True)
    staged = []
    for i, src in enumerate(p for p in faces if p and os.path.isfile(p)):
        base = re.sub(r"^\d+_", "", os.path.basename(src))
        dst = os.path.join(tmp_dir, f"{i + 1:02d}_{base}")
        shutil.copy2(src, dst)  # copy first: sources may be inside faces/
        staged.append(dst)
    for f in os.listdir(faces_dir_abs):
        os.remove(os.path.join(faces_dir_abs, f))
    for path in staged:
        shutil.move(path, os.path.join(faces_dir_abs, os.path.basename(path)))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    voice = {k: voice_settings.get(k) for k in VOICE_KEYS if k in voice_settings}
    for k in ("model_file", "index_file"):
        if k in voice:
            voice[k] = _portable(voice[k])
    data = {
        "name": str(name).strip(),
        "face": {k: face_settings.get(k) for k in FACE_KEYS if k in face_settings},
        "voice": voice,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(d, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return load_profile(name)


def delete_profile(name):
    """Delete a profile folder (only ever inside PROFILES_DIR)."""
    d = os.path.abspath(_dir(name))
    if os.path.dirname(d) != os.path.abspath(PROFILES_DIR):
        raise ValueError("Refusing to delete outside the profiles folder.")
    if os.path.isdir(d):
        shutil.rmtree(d)


def match_choice(stored_path, choices):
    """The dropdown value in ``choices`` that is the same file as
    ``stored_path`` (paths differ in separators between Mac and Windows)."""
    if not stored_path:
        return None
    want = os.path.normcase(os.path.normpath(stored_path.replace("/", os.sep)))
    for value in choices:
        if os.path.normcase(os.path.normpath(str(value))) == want:
            return value
    return stored_path
