import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from rvc.lib.tools.http_download import download_http

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt", "swift.onnx"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}


def is_downloaded(file):
    return os.path.isfile(file) and os.path.getsize(file) > 0


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not is_downloaded(destination_path):
                url = f"{url_base}/{remote_folder}{file}"
                # Size discovery only controls progress; it must never prevent
                # downloading when HEAD is unsupported or has no Content-Length.
                try:
                    with requests.head(
                        url, allow_redirects=True, timeout=30
                    ) as response:
                        response.raise_for_status()
                        total_size += int(response.headers.get("content-length", 0))
                except (requests.RequestException, ValueError):
                    pass
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    previous = 0

    def progress(received, total):
        nonlocal previous
        global_bar.update(received - previous)
        previous = received

    download_http(url, destination_path, progress=progress)


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not is_downloaded(destination_path):
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def calculate_total_size(models, exe):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    return total_size


def prequisites_download_pipeline(models, exe):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(models, exe)

    with tqdm(
        total=total_size or None,
        unit="iB",
        unit_scale=True,
        desc="Downloading all files",
    ) as global_bar:
        if models:
            download_mapping_files(models_list, global_bar)
            download_mapping_files(embedders_list, global_bar)
        if exe:
            if os.name == "nt":
                download_mapping_files(executables_list, global_bar)
            else:
                print("No executables needed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Applio Prerequisites Downloader")
    parser.add_argument("--models", action="store_true", default=False)
    parser.add_argument("--exe", action="store_true", default=False)
    args = parser.parse_args()

    prequisites_download_pipeline(args.models, args.exe)
    print("Prerequisites installed successfully.")
