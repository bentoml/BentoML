import atexit
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
import logging
import requests
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


def _get_command():
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
        raise BentoMLException("{system} is not supported".format(system=system))
    return command


def _run_ngrok(port):
    command = _get_command()
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    _download_ngrok(ngrok_path)
    executable = str(Path(ngrok_path, command))
    os.chmod(executable, 0o777)
    ngrok = subprocess.Popen([executable, 'http', str(port)])
    atexit.register(ngrok.terminate)
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    time.sleep(1)
    tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
    j = json.loads(tunnel_url)

    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
    tunnel_url = tunnel_url.replace("https", "http")
    return tunnel_url


def _download_ngrok(ngrok_path):
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
    download_path = _download_file(url)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(ngrok_path)


def _download_file(url):
    """
    Download ngrok binary file to local
    :param url:
    :return:
    """
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(tempfile.gettempdir(), local_filename))
    with open(download_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return download_path


def start_ngrok(port):
    """
    Start ngrok server
    """
    ngrok_address = _run_ngrok(port)
    logger.info(f" * Running on {ngrok_address}")
    logger.info(" * Traffic stats available on http://127.0.0.1:4040")
