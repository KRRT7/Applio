import json
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
elif sys.platform == "win32":
    # Ensure virtualenv Scripts and torch DLL directories are accessible
    # for dynamic loading of ROCm and runtime libraries
    scripts_dir = os.path.dirname(sys.executable)
    if os.path.isdir(scripts_dir):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(scripts_dir)
            except Exception:
                pass
        curr_path = os.environ.get("PATH", "")
        if scripts_dir.lower() not in curr_path.lower():
            os.environ["PATH"] = f"{scripts_dir};{curr_path}"

    venv_base = os.path.dirname(scripts_dir)
    torch_lib = os.path.join(venv_base, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
        curr_path = os.environ.get("PATH", "")
        if torch_lib.lower() not in curr_path.lower():
            os.environ["PATH"] = f"{torch_lib};{curr_path}"

    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ.setdefault("MIOPEN_DEBUG_DISABLE_FIND_DB", "1")
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
    os.environ.setdefault("AMD_COMGR_CACHE", "0")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

current_script_directory = os.path.dirname(os.path.realpath(__file__))
if current_script_directory not in sys.path:
    sys.path.append(current_script_directory)


def run_prerequisites_script(models: bool = True, exe: bool = True):
    try:
        from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

        prequisites_download_pipeline(models, exe)
        return "Prerequisites downloaded and installed successfully."
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. Please install requirements: uv sync"
        )


import click


def _get_version():
    config_path = os.path.join(
        current_script_directory, "assets", "config_template.json"
    )
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


VERSION = _get_version()


@click.group(invoke_without_command=True)
@click.version_option(
    version=VERSION, prog_name="Applio", message="%(prog)s v%(version)s"
)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option("--models/--no-models", default=True, help="Download additional models.")
@click.option("--exe/--no-exe", default=True, help="Download required executables.")
def prerequisites(**kwargs):
    """Install prerequisites for RVC."""
    result = run_prerequisites_script(kwargs["models"], kwargs["exe"])
    click.echo(result)


def main():
    cli()


if __name__ == "__main__":
    main()
