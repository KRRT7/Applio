"""Pick sensible live-voice audio devices and check the routing.

The usual setup is: your mic -> Applio -> a virtual cable's playback side,
which other apps (Discord, OBS, Zoom, WhatsApp Web) read as a microphone
from the cable's recording side. On Windows that is VB-Audio's "CABLE
Input" -> "CABLE Output"; on macOS it is BlackHole, one device with both
sides ("BlackHole 2ch"). Nothing on a virtual cable reaches your speakers,
so to hear yourself a monitor device (your speakers/headphones) is needed
as well.

The selection logic here is pure (works on device records) so it can be
tested without audio hardware; `query()` and `test_cable()` touch PortAudio.
"""

import re
import sys
from dataclasses import dataclass, field

import numpy as np

# Host APIs in order of preference. WASAPI shares the device with other apps,
# has full device names (MME truncates them to 31 chars) and low latency.
# WDM-KS is left out: it is exclusive and often refuses formats.
HOST_PREFERENCE = ("WASAPI", "MME", "DirectSound", "Core Audio", "ALSA", "JACK")

_VIRTUAL_WORDS = (
    "vb-audio", "voicemeeter", "virtual", "blackhole", "loopback", "stereo mix",
    # macOS wrappers: "Applio Voice" is an aggregate around BlackHole for call
    # apps that hide BlackHole itself, so it carries Applio's own output;
    # Audio MIDI Setup names new ones "Aggregate Device" / "Multi-Output Device".
    "applio voice", "aggregate device", "multi-output device",
)


def _is_mac(mac=None):
    return sys.platform == "darwin" if mac is None else mac


def cable_name(mac=None):
    """The virtual cable's playback side to suggest, per platform."""
    return "BlackHole 2ch" if _is_mac(mac) else "CABLE Input"


def mic_access_hint(mac=None):
    return ("System Settings → Privacy & Security → Microphone" if _is_mac(mac)
            else "Windows Settings → Privacy → Microphone")


def _channels(name):
    """'BlackHole 16ch' -> 16, 'CABLE In 16ch (...)' -> 16, no count -> 0."""
    m = re.search(r"(\d+)\s*ch\b", name.lower())
    return int(m.group(1)) if m else 0


@dataclass
class Device:
    index: int
    name: str
    host_api: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: float = 48000.0


@dataclass
class Setup:
    input: Device = None
    output: Device = None
    monitor: Device = None
    notes: list = field(default_factory=list)


def is_virtual(name):
    n = (name or "").lower()
    return any(w in n for w in _VIRTUAL_WORDS) or n.startswith("cable ")


def is_cable_playback(name):
    """A virtual cable's playback side, i.e. where Applio should send audio."""
    n = (name or "").lower()
    return bool(
        re.match(r"cable(-\w+)? in(put)?\b", n)
        or ("voicemeeter" in n and "input" in n)
        or n.startswith("blackhole")
    )


def cable_capture_name(output_name):
    """Name prefix of the recording side that other apps pick as their mic,
    for a cable playback device name. 'CABLE Input (...)' -> 'CABLE Output'."""
    name = output_name or ""
    m = re.match(r"(CABLE(?:-\w+)?) In", name, re.IGNORECASE)
    if m:
        return f"{m.group(1)} Output"
    m = re.match(r"(.*?)\bInput\b", name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}Output"
    if name.lower().startswith("blackhole"):
        return name.split(" (")[0]
    return None


def _host_rank(device):
    for rank, host in enumerate(HOST_PREFERENCE):
        if host.lower() in device.host_api.lower():
            return rank
    return len(HOST_PREFERENCE)


def _best_host(devices):
    """Devices on the most preferred host API that has any."""
    if not devices:
        return []
    best = min(_host_rank(d) for d in devices)
    return [d for d in devices if _host_rank(d) == best]


def _same_endpoint(a, b):
    """MME truncates names to 31 chars, so compare by prefix."""
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def pick(inputs, outputs, default_input=None, default_output=None, mac=None):
    """Choose input, output and monitor devices.

    inputs/outputs: Device lists. default_input/default_output: the system
    default endpoints (Device or None), used to pick the mic and speakers.
    """
    setup = Setup()

    # Output: a virtual cable's playback side. Prefer the plain stereo one
    # ('CABLE Input' over 'CABLE In 16ch', 'BlackHole 2ch' over 16ch/64ch).
    cables = _best_host([d for d in outputs if is_cable_playback(d.name)])
    cables.sort(key=lambda d: (_channels(d.name), d.index))
    if cables:
        setup.output = cables[0]
    elif _is_mac(mac):
        setup.notes.append(
            "No virtual cable found. Install BlackHole 2ch (existential.audio/blackhole, "
            "or: brew install blackhole-2ch), restart Applio, then press Auto-setup again."
        )
    else:
        setup.notes.append(
            "No virtual cable found. Install VB-Audio Virtual Cable (vb-audio.com/Cable), "
            "restart Applio, then press Auto-setup again."
        )

    # Input: the system default mic, as long as it is a real one.
    real_inputs = [d for d in inputs if not is_virtual(d.name) and "sound mapper" not in d.name.lower()
                   and "primary sound" not in d.name.lower()]
    candidates = _best_host(real_inputs)
    if default_input is not None and not is_virtual(default_input.name):
        same = [d for d in candidates if _same_endpoint(d.name, default_input.name)]
        candidates = same or candidates
    if candidates:
        setup.input = candidates[0]
    else:
        setup.notes.append(f"No microphone found. Plug one in and allow microphone access in {mic_access_hint(mac)}.")

    # Monitor device: the system default speakers/headphones. Only filled
    # in; whether to use it is the user's choice.
    real_outputs = [d for d in outputs if not is_virtual(d.name) and "sound mapper" not in d.name.lower()
                    and "primary sound" not in d.name.lower()]
    candidates = _best_host(real_outputs)
    if default_output is not None and not is_virtual(default_output.name):
        same = [d for d in candidates if _same_endpoint(d.name, default_output.name)]
        candidates = same or candidates
    if candidates and setup.output is not None:
        setup.monitor = candidates[0]
    elif setup.output is None and candidates:
        # Without a cable, send straight to the speakers so there is sound at all.
        setup.output = candidates[0]

    return setup


# ─── label helpers (the realtime dropdowns use "N: name (host api)") ─────


_LABEL = re.compile(r"^\d+: (.*) \(([^()]*)\)$")


def parse_label(label):
    """'6: CABLE Input (VB-Audio Virtual Cable) (Windows WASAPI)' -> (name, host)."""
    m = _LABEL.match(label or "")
    return (m.group(1), m.group(2)) if m else ((label or ""), "")


def label_for(device, labels):
    """The dropdown label (from {label: index}) for a Device, or None."""
    if device is None:
        return None
    for label, index in labels.items():
        if index == device.index:
            return label
    return None


# ─── checks shown in the Live page's audio panel ─────────────────────────

OK, WARN, BAD, OFF = "ok", "warn", "bad", "off"


def check(input_label, output_label, monitor_label, use_monitor, mac=None):
    """Per-step verdicts for the routing: dict of step -> (level, text)."""
    in_name, _ = parse_label(input_label)
    out_name, _ = parse_label(output_label)
    mon_name, _ = parse_label(monitor_label)
    steps = {}

    if not input_label:
        steps["mic"] = (BAD, "No microphone selected")
    elif is_virtual(in_name):
        steps["mic"] = (BAD, "This is a virtual device, not a microphone: Applio would hear nothing (or itself)")
    else:
        steps["mic"] = (OK, in_name)

    capture = cable_capture_name(out_name) if is_cable_playback(out_name) else None
    if not output_label:
        steps["out"] = (BAD, "No output selected")
    elif capture:
        apps = "WhatsApp Web/Zoom/OBS" if _is_mac(mac) else "Discord/OBS/Zoom"
        steps["out"] = (OK, f'{out_name} → in {apps} pick "{capture}" as the microphone')
    elif is_virtual(out_name):
        steps["out"] = (WARN, f"{out_name}: this looks like a recording device, not a cable's playback side")
    else:
        steps["out"] = (WARN, f"{out_name}: other apps can't use this as a mic. "
                              f"Pick {cable_name(mac)} to send your voice to calls")

    if use_monitor and monitor_label:
        if mon_name == out_name:
            steps["hear"] = (WARN, "Monitor is the same device as the output")
        elif is_virtual(mon_name):
            steps["hear"] = (WARN, f"{mon_name} is virtual, so you still won't hear yourself")
        else:
            steps["hear"] = (OK, mon_name)
    elif capture:
        steps["hear"] = (OFF, "Off: you won't hear yourself (the cable is silent by design). "
                              "To listen, tick Use Monitor Device (Voice tab → Audio Devices → Monitor Device)")
    else:
        steps["hear"] = (OK, f"Through the output ({out_name})" if output_label else "Nothing")
    return steps


# ─── PortAudio access ────────────────────────────────────────────────────


def query():
    """(inputs, outputs, default_input, default_output) from PortAudio."""
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    devices = []
    for d in sd.query_devices():
        devices.append(Device(d["index"], d["name"], hostapis[d["hostapi"]]["name"],
                              d["max_input_channels"], d["max_output_channels"], d["default_samplerate"]))
    by_index = {d.index: d for d in devices}

    # The default endpoints as the most preferred host API reports them
    # (WASAPI's defaults are the real Windows defaults, with full names).
    default_in = default_out = None
    for host in HOST_PREFERENCE:
        api = next((h for h in hostapis if host.lower() in h["name"].lower()), None)
        if api is None:
            continue
        default_in = default_in or by_index.get(api.get("default_input_device", -1))
        default_out = default_out or by_index.get(api.get("default_output_device", -1))
        if default_in and default_out:
            break

    inputs = [d for d in devices if d.max_input_channels > 0]
    outputs = [d for d in devices if d.max_output_channels > 0]
    return inputs, outputs, default_in, default_out


def find_capture(output_device, inputs):
    """The recording side of a cable playback device, on the same host API if possible."""
    prefix = cable_capture_name(output_device.name)
    if not prefix:
        return None
    matches = [d for d in inputs if d.name.lower().startswith(prefix.lower()[:31])]
    same_host = [d for d in matches if d.host_api == output_device.host_api]
    return (same_host or matches or [None])[0]


def test_cable(output_index, seconds=0.6, level=0.1):
    """Play a short quiet tone into a cable and record it back from the
    cable's recording side. Returns (ok, message)."""
    import sounddevice as sd

    inputs, outputs, _, _ = query()
    out = next((d for d in outputs if d.index == output_index), None)
    if out is None:
        return False, "The output device was not found. Press Refresh Audio Devices."
    if not is_cable_playback(out.name):
        return False, f"{out.name} is not a virtual cable, so there is nothing to loop back."
    cap = find_capture(out, inputs)
    if cap is None:
        return False, f'Could not find "{cable_capture_name(out.name)}" to record from.'

    sr = int(out.default_samplerate)
    wasapi = "WASAPI" in out.host_api
    extra_out = sd.WasapiSettings(auto_convert=True) if wasapi else None
    extra_in = sd.WasapiSettings(auto_convert=True) if "WASAPI" in cap.host_api else None
    out_ch = min(2, out.max_output_channels)
    in_ch = min(2, cap.max_input_channels)

    n = int(sr * seconds)
    tone = (level * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)).astype(np.float32)
    pos = [0]
    recorded = []

    def play(outdata, frames, _time, _status):
        chunk = tone[pos[0]:pos[0] + frames]
        outdata.fill(0)
        outdata[:len(chunk)] = chunk[:, None]
        pos[0] += frames

    def record(indata, _frames, _time, _status):
        recorded.append(indata.copy())

    def duplex(indata, outdata, frames, time_, status):
        play(outdata, frames, time_, status)
        record(indata, frames, time_, status)

    try:
        if cap.index == out.index:
            # BlackHole on macOS: one device with both sides. Core Audio
            # refuses a second stream on it, so play and record in one.
            with sd.Stream(device=out.index, channels=(in_ch, out_ch), samplerate=sr,
                           dtype="float32", callback=duplex):
                sd.sleep(int(seconds * 1000) + 300)
        else:
            with sd.InputStream(device=cap.index, channels=in_ch, samplerate=sr, dtype="float32",
                                callback=record, extra_settings=extra_in):
                with sd.OutputStream(device=out.index, channels=out_ch, samplerate=sr, dtype="float32",
                                     callback=play, extra_settings=extra_out):
                    sd.sleep(int(seconds * 1000) + 300)
    except Exception as exc:
        return False, f"Could not open the cable: {exc}"

    if not recorded:
        return False, "No audio came back from the cable."
    peak = float(np.max(np.abs(np.concatenate(recorded))))
    if peak < level * 0.2:
        where = ("Audio MIDI Setup (Applications → Utilities)" if _is_mac()
                 else "Windows Sound settings (Recording tab)")
        return False, (f'Played a tone into {out.name} but "{cap.name}" stayed silent. '
                       f"Check the cable isn't muted in {where}.")
    return True, f'Cable works: a test tone sent to {out.name} came back on "{cap.name}".'
