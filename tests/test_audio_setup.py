"""Tests for live-voice audio auto-setup and the Live page's audio panel.

Device lists mirror a real Windows machine with VB-Audio Virtual Cable;
no audio hardware is touched.
"""
import types

from rvc.realtime import audio_setup as aus
from rvc.realtime.audio_setup import Device
from tabs.faceswap import faceswap as fs

MME, WAS = "MME", "Windows WASAPI"
INPUTS = [
    Device(0, "Microsoft Sound Mapper - Input", MME, 2, 0),
    Device(1, "Microphone Array (Intel® Smart ", MME, 4, 0),
    Device(2, "CABLE Output (VB-Audio Virtual ", MME, 16, 0),
    Device(17, "CABLE Output (VB-Audio Virtual Cable)", WAS, 2, 0),
    Device(18, "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)", WAS, 2, 0),
    Device(22, "Stereo Mix (Realtek HD Audio Stereo input)", "Windows WDM-KS", 2, 0),
]
OUTPUTS = [
    Device(3, "Microsoft Sound Mapper - Output", MME, 0, 2),
    Device(5, "CABLE In 16ch (VB-Audio Virtual", MME, 0, 16),
    Device(6, "CABLE Input (VB-Audio Virtual C", MME, 0, 16),
    Device(14, "CABLE In 16ch (VB-Audio Virtual Cable)", WAS, 0, 2),
    Device(15, "Speakers (Realtek(R) Audio)", WAS, 0, 2),
    Device(16, "CABLE Input (VB-Audio Virtual Cable)", WAS, 0, 2),
]


def test_pick_routes_mic_to_cable_and_monitors_on_speakers():
    s = aus.pick(INPUTS, OUTPUTS, default_input=INPUTS[4], default_output=OUTPUTS[4])
    assert s.input.index == 18          # the real mic, on WASAPI
    assert s.output.index == 16         # plain stereo CABLE Input, not 16ch, not MME
    assert s.monitor.index == 15        # speakers, filled in for the monitor
    assert s.notes == []


def test_pick_ignores_a_virtual_default_mic():
    s = aus.pick(INPUTS, OUTPUTS, default_input=INPUTS[3], default_output=OUTPUTS[5])
    assert s.input.index == 18
    assert s.monitor.index == 15


def test_pick_without_a_cable_plays_to_the_speakers_and_says_why():
    outputs = [d for d in OUTPUTS if not aus.is_virtual(d.name)]
    s = aus.pick(INPUTS, outputs, default_input=INPUTS[4], default_output=OUTPUTS[4])
    assert s.output.index == 15 and s.monitor is None
    assert any("VB-Audio" in n for n in s.notes)


def test_cable_names():
    assert aus.is_cable_playback("CABLE Input (VB-Audio Virtual Cable)")
    assert aus.is_cable_playback("CABLE In 16ch (VB-Audio Virtual Cable)")
    assert aus.is_cable_playback("CABLE-A Input (VB-Audio Cable A)")
    assert not aus.is_cable_playback("CABLE Output (VB-Audio Virtual Cable)")
    assert not aus.is_cable_playback("Speakers (Realtek(R) Audio)")
    assert aus.cable_capture_name("CABLE Input (VB-Audio Virtual Cable)") == "CABLE Output"
    assert aus.cable_capture_name("CABLE In 16ch (VB-Audio Virtual Cable)") == "CABLE Output"
    assert aus.cable_capture_name("CABLE-A Input (VB-Audio Cable A)") == "CABLE-A Output"
    assert aus.cable_capture_name("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)") == "VoiceMeeter Output"


def test_find_capture_prefers_same_host_api():
    assert aus.find_capture(OUTPUTS[5], INPUTS).index == 17
    assert aus.find_capture(OUTPUTS[2], INPUTS).index == 2   # MME names are truncated


def test_labels_round_trip():
    labels = {"1: Speakers (Realtek(R) Audio) (Windows WASAPI)": 15,
              "2: CABLE Input (VB-Audio Virtual Cable) (Windows WASAPI)": 16}
    assert aus.parse_label("2: CABLE Input (VB-Audio Virtual Cable) (Windows WASAPI)") == (
        "CABLE Input (VB-Audio Virtual Cable)", "Windows WASAPI")
    assert aus.label_for(OUTPUTS[5], labels).startswith("2: CABLE Input")
    assert aus.label_for(None, labels) is None


MIC = "5: Microphone Array (Intel® Smart Sound Technology for Digital Microphones) (Windows WASAPI)"
CABLE = "6: CABLE Input (VB-Audio Virtual Cable) (Windows WASAPI)"
SPK = "8: Speakers (Realtek(R) Audio) (Windows WASAPI)"


def test_check_flags_the_no_sound_setup():
    """The reported bug: output on the cable, no monitor -> silent by design."""
    steps = aus.check(MIC, CABLE, SPK, use_monitor=False)
    assert steps["mic"][0] == aus.OK
    assert steps["out"][0] == aus.OK and '"CABLE Output"' in steps["out"][1]
    assert steps["hear"][0] == aus.OFF and "won't hear" in steps["hear"][1]
    assert aus.check(MIC, CABLE, SPK, use_monitor=True)["hear"] == (aus.OK, "Speakers (Realtek(R) Audio)")


def test_check_flags_a_cable_picked_as_the_mic():
    steps = aus.check("3: CABLE Output (VB-Audio Virtual Cable) (Windows WASAPI)", CABLE, SPK, True)
    assert steps["mic"][0] == aus.BAD


def test_check_warns_when_output_is_speakers():
    assert aus.check(MIC, SPK, None, False)["out"][0] == aus.WARN


def _with_rt(monkeypatch, stats=None, running=False, status=None):
    rt = types.SimpleNamespace(running=running, SESSION_CONTROLS={})
    rt.voice_stats = lambda: stats
    rt.voice_status = lambda: status
    monkeypatch.setattr(fs, "_realtime", lambda: rt)
    fs._VOICE_WATCH.clear()
    return rt


def test_panel_ready_and_not_ready(monkeypatch):
    _with_rt(monkeypatch)
    out = fs.set_audio_config(MIC, CABLE, SPK, True, "logs/me/me.pth")
    assert "Ready" in out and "Speakers" in out
    assert "terms" not in out and ">Voice<" not in out   # no Voice row, no terms step
    out = fs.set_audio_config(MIC, CABLE, SPK, True, "")
    assert "Not ready" in out and "No voice model" in out


def test_panel_shows_meters_while_live(monkeypatch):
    stats = {"blocks": 10, "input_db": -20.0, "output_db": float("-inf")}
    _with_rt(monkeypatch, stats=stats, running=True)
    out = fs.set_audio_config(MIC, CABLE, SPK, True, "me.pth")
    assert "Live" in out and "-20 dB" in out and "silent" in out


def test_panel_tick_skips_when_unchanged(monkeypatch):
    _with_rt(monkeypatch)
    fs.set_audio_config(MIC, CABLE, SPK, True, "me.pth")
    assert not isinstance(fs._audio_update(), str)   # gr.skip()


def test_changing_output_clears_the_cable_test(monkeypatch):
    _with_rt(monkeypatch)
    fs.set_audio_config(MIC, CABLE, SPK, True, "me.pth")
    fs._AUDIO["test"] = (True, "Cable works")
    assert "Cable works" in fs.set_audio_config(MIC, CABLE, SPK, False, "me.pth")
    assert "Cable works" not in fs.set_audio_config(MIC, SPK, SPK, False, "me.pth")
