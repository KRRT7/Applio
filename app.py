# Make sure the config file exists
import os
import shutil
import sys

# Nuitka-compiled build: run from the unpacked program folder and keep user data
# in a persistent per-user folder (see assets/compiled_runtime.py).
if "__compiled__" in globals():
    from assets.compiled_runtime import prepare

    prepare(os.path.dirname(os.path.abspath(__file__)))

# We need the CWD for finding the config file, but while we're at it, add it to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.updater import ensure_not_updating

ensure_not_updating(now_dir)

# TODO: This path is regenerated all over the place in Applio
# should probably be in a static module for everything to reference
CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
os.environ.setdefault("APPLIO_CONFIG_FILE", CONFIG_PATH)

# The base config file to start from
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")

if not os.path.exists(CONFIG_PATH):
    print("Config file not found. Creating fresh from template.")
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

# Plataform config
from rvc.lib.platform import platform_config

platform_config()

import argparse
import types
import gradio as gr
import pathlib
import logging

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

_ARG_PARSER = argparse.ArgumentParser(
    description="Applio Web UI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_ARG_PARSER.add_argument(
    "--port", type=int, default=DEFAULT_PORT, help="Server port (default: %(default)s)"
)
_ARG_PARSER.add_argument(
    "--server-name",
    type=str,
    default=DEFAULT_SERVER_NAME,
    help="Server hostname (default: %(default)s)",
)
_ARG_PARSER.add_argument(
    "--share", action="store_true", help="Create a public Gradio share link"
)
_ARG_PARSER.add_argument(
    "--open", action="store_true", help="Open the browser automatically"
)
_ARG_PARSER.add_argument(
    "--client", action="store_true", help="Enable client mode (mounts realtime API)"
)
_args, _ = _ARG_PARSER.parse_known_args()
client_mode = _args.client
_has_share = _args.share
_has_open = _args.open

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress ConnectionResetError on Windows when a remote peer forcibly closes the
# connection during asyncio shutdown (WinError 10054 / ProactorBasePipeTransport).
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

    def _ccl_patched(self, exc):
        try:
            _orig_ccl(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched

# Fix Gradio NoneType error when entering an invalid value
gr.Number.preprocess = types.MethodType(
    lambda self, payload: (
        None
        if payload is None
        or (self.minimum is not None and payload < self.minimum)
        or (self.maximum is not None and payload > self.maximum)
        else self.round_to_precision(payload, self.precision)
    ),
    gr.Number,
)

# detect gradio
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from assets.app_shell import build_workspace, create_theme, stylesheet
from tabs.settings.settings import settings_tab
from tabs.realtime.realtime import realtime_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = (
    create_theme()
    if loadThemes.read_current_theme() in ("Applio", "ParityError/Interstellar")
    else loadThemes.load_theme() or create_theme()
)
app_css = stylesheet()

# Define Gradio interface
with gr.Blocks(
    title="Applio",
    **(
        {
            "theme": my_applio,
            "css": app_css,
            "js": (
                (
                    "() => {\n"
                    + pathlib.Path(
                        os.path.join(now_dir, "tabs", "realtime", "main.js")
                    ).read_text()
                    + "\n}"
                )
                if client_mode
                else None
            ),
        }
        if not GRADIO_6
        else {}
    ),
) as Applio:
    build_workspace(
        {
            "realtime": realtime_tab,
            "settings": settings_tab,
        },
        i18n,
    )


def launch_gradio(server_name: str, server_port: int) -> None:
    app, _, _ = Applio.launch(
        favicon_path="assets/ICON.ico",
        share=_has_share,
        inbrowser=_has_open,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=client_mode,
        allowed_paths=["logs"],
        **(
            {
                "theme": my_applio,
                "css": app_css,
                "js": (
                    pathlib.Path(
                        os.path.join(now_dir, "tabs", "realtime", "main.js")
                    ).read_text()
                    if client_mode
                    else None
                ),
            }
            if GRADIO_6
            else {}
        ),
    )

    if client_mode:
        import time
        from rvc.realtime.client import app as fastapi_app

        app.mount("/api", fastapi_app)

        while True:
            time.sleep(5)


if __name__ == "__main__":
    port = _args.port
    server = _args.server_name

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
