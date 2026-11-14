import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()

from tabs.settings.sections.realtime_audio import realtime_audio_tab
from tabs.settings.sections.themes import theme_tab
from tabs.settings.sections.version import version_tab
from tabs.settings.sections.lang import lang_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.rmvpe_high_register import rmvpe_high_register_tab


def settings_tab():
    with gr.TabItem(label=i18n("General")):
        with step_card(
            1,
            "Pitch Correction",
            i18n,
            description="Fix high-register octave errors in RMVPE pitch.",
        ):
            rmvpe_high_register_tab()
        with step_card(
            2,
            "Audio Devices",
            i18n,
            description="Configure drivers and the streaming sample rate.",
        ):
            realtime_audio_tab()
        with step_card(
            3,
            "Appearance & Language",
            i18n,
            description="Pick a theme and language. Applies after restart.",
        ):
            theme_tab()
            lang_tab()
        with step_card(
            4,
            "Updates & System",
            i18n,
            description="Check for updates or restart the interface.",
        ):
            version_tab()
            restart_tab()
