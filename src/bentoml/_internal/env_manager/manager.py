from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from shutil import which
from tempfile import NamedTemporaryFile

import fs
import attr

from ...exceptions import BentoMLException
from ..configuration import get_debug_mode

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from fs.base import FS


def decode(msg: bytes) -> str:
    if msg:
        return msg.decode("utf-8")
    return ""


def get_python_version_from_bento(bento_path: str) -> str:
    with open(
        os.path.join(bento_path, "env", "python", "version.txt"), "r"
    ) as pyver_file:
        py_version = pyver_file.read().split(".")[:2]
        return ".".join(py_version)


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
        safer_bash_args = [
            "-euxo",
            "pipefail",
        ]
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


@attr.define
class EnvManager:
    env_name: str
    env_type: str
    is_ephemeral: bool
    bento_path: t.Optional[str] = None
    bentofile_path: t.Optional[str] = None
    _env_fs: FS = attr.field(init=False)

    def __attrs_post_init__(self):
        from bentoml._internal.configuration.containers import BentoMLContainer

        env_home = fs.open_fs(BentoMLContainer.env_store_dir.get())

        if not self.is_ephemeral:
            assert (
                self.env_name is not None
            ), "persistent environments need a valid name."

        assert not (
            self.bento_path is None and self.bentofile_path is None
        ), "Provide bentopath or path to bentofile."

        if self.env_type == "conda":
            if not env_home.exists("conda"):
                env_home.makedir("conda")
            self._env_fs = (
                fs.open_fs("temp://")
                if self.is_ephemeral
                else env_home.opendir("conda")
            )
            self.create_conda_env()

    @classmethod
    def from_bento(
        cls,
        env_type: str,
        is_ephemeral: bool,
        env_name: t.Optional[str] = None,
        bento_path: t.Optional[str] = None,
    ) -> EnvManager:
        if env_name is None:
            env_name = "ephemeral_env"
        return cls(
            env_name=env_name,
            env_type=env_type,
            is_ephemeral=is_ephemeral,
            bento_path=bento_path,
        )

    @classmethod
    def from_bentofile(cls) -> EnvManager:
        raise NotImplementedError

    def create_conda_env(self) -> str:
        """
        Create a new conda env with self.env_name
        """
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe is None:
            raise BentoMLException(
                "conda executable not found! Make sure conda is installed and that `CONDA_EXE` is set."
            )
        # create a env under $BENTOML_HOME/env
        # setup conda with bento's environment.yml file and python/install.sh file
        conda_env_path: str
        if self._env_fs.exists(self.env_name):
            return self._env_fs.getsyspath(self.env_name)
        else:
            assert self.bento_path is not None, "bento_path not provided."
            with NamedTemporaryFile(mode="w", delete=False) as script_file:
                conda_env_path = self._env_fs.getsyspath(self.env_name)
                python_version = get_python_version_from_bento(self.bento_path)
                script_file.write(
                    f"conda create -p {conda_env_path} python={python_version} --yes"
                    + "\n"
                )

                # install conda deps
                from ..bento.build_config import CONDA_ENV_YAML_FILE_NAME

                conda_environment_file = fs.path.join(
                    self.bento_path, "env", "conda", CONDA_ENV_YAML_FILE_NAME
                )
                if os.path.exists(conda_environment_file):
                    script_file.write(
                        "conda config --set pip_interop_enabled True" + "\n"
                    )
                    script_file.write(
                        f"conda env update -p {conda_env_path} --file {conda_environment_file}"
                        + "\n"
                    )

                script_file.write(f'eval "$(conda shell.posix hook)"' + "\n")
                script_file.write(f"conda activate {conda_env_path}" + "\n")

                python_install_script = fs.path.join(
                    self.bento_path, "env", "python", "install.sh"
                )
                script_file.write(f"bash -euxo pipefail {python_install_script}" + "\n")

            logger.info("Creating Conda env and installing dependencies...")
            run_script_subprocess(
                script_file.name,
                capture_output=get_debug_mode(),
                debug_mode=get_debug_mode(),
            )
            return conda_env_path

    def run(self, commands: list[str]):
        """
        Run commands in the activated environment.
        """
        with NamedTemporaryFile(mode="w", delete=False) as script_file:
            conda_env_path = self._env_fs.getsyspath(self.env_name)
            script_file.write(f'eval "$(conda shell.posix hook)"' + "\n")
            script_file.write(f"conda activate {conda_env_path}" + "\n")
            script_file.write(" ".join(commands) + "\n")
        run_script_subprocess(
            script_file.name, capture_output=False, debug_mode=get_debug_mode()
        )
