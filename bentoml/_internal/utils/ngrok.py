import os
import json
import time
import shutil
import logging
import zipfile
import platform
import tempfile
import subprocess
from pathlib import Path
from threading import Thread

import requests

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


def get_command() -> str:
    """
    ngrok command based on OS
    """
    system = platform.system()
    if system == "Darwin":
        command = "ngrok"
    elif system == "Windows":
        command = "ngrok.exe"
    elif system == "Linux":
        command = "ngrok"
    else:
        raise BentoMLException(f"{system} is not supported")
    return command


def log_url() -> None:
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    while True:
        time.sleep(1)
        response = requests.get(localhost_url)
        if response.status_code == 200:
            data = json.loads(response.text)
            if data["tunnels"]:
                tunnel = data["tunnels"][0]
                logger.info(
                    " Ngrok running at: %s",
                    tunnel["public_url"].replace("https://", "http://"),
                )
                logger.info(" Traffic stats available on http://127.0.0.1:4040")
                return
        else:
            logger.info("Waiting for ngrok to start...")


def start_ngrok(port: int):
    """
    Start ngrok server synchronously
    """
    command = get_command()
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    download_ngrok(ngrok_path)
    executable = str(Path(ngrok_path, command))
    os.chmod(executable, 0o777)
    Thread(target=log_url).start()
    with subprocess.Popen([executable, "http", str(port)]) as ngrok_process:
        ngrok_process.wait()


def download_ngrok(ngrok_path: str) -> None:
    """
    Check OS and decide on ngrok download URL
    """
    if Path(ngrok_path).exists():
        return
    system = platform.system()
    if system == "Darwin":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip"
    elif system == "Windows":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip"
    elif system == "Linux":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"
    else:
        raise Exception(f"{system} is not supported")
    download_path = download_file(url)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(ngrok_path)


def download_file(url: str) -> str:
    """
    Download ngrok binary file to local

    Args:
        url (:code:`str`):
            URL to download

    Returns:
        :code:`download_path`: str
    """
    local_filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(tempfile.gettempdir(), local_filename))
    with open(download_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return download_path
