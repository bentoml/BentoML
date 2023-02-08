from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from abc import ABC
from abc import abstractmethod
from shutil import which
from tempfile import NamedTemporaryFile

import attr
from fs.base import FS

from ...exceptions import BentoMLException
from ..bento.bento import Bento
from ..configuration import get_debug_mode

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from fs.base import FS


def decode(msg: bytes) -> str:
    if msg:
        return msg.decode("utf-8")
    return ""


@attr.define
class Environment(ABC):
    name: str
    env_fs: FS
    # path to bento's /env dir
    bento: Bento
    env_exe: str = attr.field(init=False)

    def __attrs_post_init__(self):
        self.env_exe = self.get_executable()
        if self.env_fs.exists(self.name):
            return self.env_fs.getsyspath(self.name)
        else:
            self.create()

    @abstractmethod
    def get_executable(self) -> str:
        """
        Get path to executable that will be used to create/manage the environment
        """
        ...

    @abstractmethod
    def create(self):
        """
        Create the environment with the files from bento.
        """
        ...

    @abstractmethod
    def run(self, commands: list[str]):
        """
        run the commands in an activated environment.
        """
        ...

    @staticmethod
    def run_script_subprocess(
        script_file_path: t.Union[str, os.PathLike[str]],
        capture_output: bool,
        debug_mode: bool,
    ):
        shell_path = which("bash")
        if shell_path is None:
            raise BentoMLException("Unable to locate a valid shell")

        safer_bash_args: list[str] = []

        # but only work in bash
        if debug_mode:
            safer_bash_args = ["-euxo", "pipefail"]
        result = subprocess.run(
            [shell_path, *safer_bash_args, script_file_path],
            capture_output=capture_output,
        )
        if result.returncode != 0:
            logger.debug(decode(result.stdout))
            logger.error(decode(result.stderr))
            raise BentoMLException(
                "Subprocess call returned non-zero value. Reffer logs for more details"
            )


class Conda(Environment):
    def get_executable(self) -> str:
        conda_exe: str
        if os.environ.get("CONDA_EXE") is not None:
            conda_exe = os.environ.get("CONDA_EXE")
        else:
            raise BentoMLException(
                "Conda|Miniconda executable not found! Make sure any one is installed and environment is activated."
            )
        return conda_exe

    def create(self):
        # create a env under $BENTOML_HOME/env
        # setup conda with bento's environment.yml file and python/install.sh file
        conda_env_path = self.env_fs.getsyspath(self.name)
        script_file_commands: list[str] = []
        python_version: str
        with open(self.bento.path_of("/env/python/version.txt"), "r") as pyver_file:
            py_version = pyver_file.read().split(".")[:2]
            python_version = ".".join(py_version)
        script_file_commands.append(
            f"conda create -p {conda_env_path} python={python_version} --yes"
        )

        # install conda deps
        from ..bento.build_config import CONDA_ENV_YAML_FILE_NAME

        conda_environment_file = self.bento.path_of(
            f"/env/conda/{CONDA_ENV_YAML_FILE_NAME}"
        )
        if os.path.exists(conda_environment_file):
            script_file_commands.append(
                f"{self.env_exe} config --set pip_interop_enabled True"
            )
            script_file_commands.append(
                f"{self.env_exe} env update -p {conda_env_path} --file {conda_environment_file}"
            )

        script_file_commands.append(f'eval "$({self.env_exe} shell.posix hook)"' + "\n")
        script_file_commands.append(f"conda activate {conda_env_path}" + "\n")

        python_install_script = self.bento.path_of("/env/python/install.sh")
        script_file_commands.append(f"bash -euxo pipefail {python_install_script}")
        with NamedTemporaryFile(mode="w", delete=False) as script_file:
            script_file.write("\n".join(script_file_commands))

        logger.info("Creating Conda env and installing dependencies...")
        self.run_script_subprocess(
            script_file.name,
            capture_output=not get_debug_mode(),
            debug_mode=get_debug_mode(),
        )

    def run(self, commands: list[str]):
        """
        Run commands in the activated environment.
        """
        conda_env_path = self.env_fs.getsyspath(self.name)
        script_file_lines: list[str] = []
        script_file_lines.append(f'eval "$({self.env_exe} shell.posix hook)"')
        script_file_lines.append(f"conda activate {conda_env_path}")
        script_file_lines.append(" ".join(commands))
        with NamedTemporaryFile(mode="w", delete=False) as script_file:
            script_file.write("\n".join(script_file_lines))
        self.run_script_subprocess(
            script_file.name, capture_output=False, debug_mode=get_debug_mode()
        )
