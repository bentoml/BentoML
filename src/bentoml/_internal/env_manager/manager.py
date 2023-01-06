from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from tempfile import NamedTemporaryFile
from distutils.spawn import find_executable as which

import fs
import attrs
from rich.status import Status

from bentoml.exceptions import BentoMLException
from bentoml._internal.configuration import get_debug_mode

logger = logging.getLogger(__name__)
is_debug_mode = get_debug_mode()

if t.TYPE_CHECKING:
    from fs.base import FS


def decode(msg: bytes) -> str:
    if msg:
        return msg.decode("utf-8")
    return ""


def run_script_subprocess(
    script_file_path: t.Union[str, os.PathLike[str]], capture_output: bool
):
    shell_path = which("bash")
    if shell_path is None:
        raise BentoMLException("Unable to locate a valid shell")

    result = subprocess.run(
        [shell_path, "-euxo", "pipefail", script_file_path],
        capture_output=capture_output,
    )
    if result.returncode != 0:
        if result.stderr and result.stdout:
            logger.debug(decode(result.stdout))
            logger.error(decode(result.stderr))
        raise BentoMLException(
            "Subprocess call returned non-zero value. Error: "
            + decode(result.stderr)
            + "\n Reffer logs for more details"
        )


@attrs.define
class EnvManager:
    env_type: str
    env_name: str
    from_bento_store: bool
    bento_path: t.Optional[str]
    _env_fs: FS = attrs.field(init=False)

    def __attrs_post_init__(self):
        env_home = fs.open_fs("/tmp/env_manager", create=True)
        if self.env_type == "conda":
            if not env_home.exists("conda"):
                env_home.makedir("conda")
            self._env_fs = env_home.opendir("conda")
            self.create_conda_env()

    @classmethod
    def get_environment(
        cls,
        env_name: str,
        env_type: str,
        from_bento_store: bool,
        bento_path: t.Optional[str] = None,
    ):
        return cls(
            env_name=env_name,
            env_type=env_type,
            from_bento_store=from_bento_store,
            bento_path=bento_path,
        )

    def create_conda_env(self) -> str:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe is None:
            raise BentoMLException(
                "conda executable not found! Make sure conda is installed and that `CONDA_EXE` is set."
            )
        # setup conda with bento's environment.yml file and python/install.sh file
        if self.from_bento_store:
            if self._env_fs.exists(self.env_name):
                return self._env_fs.getsyspath(self.env_name)
            with NamedTemporaryFile(mode="w", delete=False) as script_file:
                conda_env_path = self._env_fs.getsyspath(self.env_name)
                # TODO: figure out a proper way to get python version from a
                # bento
                python_version = "3.8"
                script_file.write(
                    f"conda create -p {conda_env_path} python={python_version} --yes"
                    + "\n"
                )

                script_file.write(f'eval "$(conda shell.posix hook)"' + "\n")
                script_file.write(f"conda activate {conda_env_path}" + "\n")
                python_install_script = fs.path.join(
                    self.bento_path, "env", "python", "install.sh"
                )
                script_file.write(f"bash -euxo pipefail {python_install_script}" + "\n")

            with Status("Creating Conda environment and installing dependencies"):
                run_script_subprocess(
                    script_file.name, capture_output=not is_debug_mode
                )
        # TODO:create an ephimeral env
        else:
            pass

    def run(self, commands: list[str]):
        """
        Run commands in the activated environment.
        """
        with NamedTemporaryFile(mode="w", delete=False) as script_file:
            conda_env_path = self._env_fs.getsyspath(self.env_name)
            script_file.write(f'eval "$(conda shell.posix hook)"' + "\n")
            script_file.write(f"conda activate {conda_env_path}" + "\n")
            script_file.write(" ".join(commands) + "\n")
        run_script_subprocess(script_file.name, capture_output=not is_debug_mode)
