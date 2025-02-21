from __future__ import annotations

import logging
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import fs

from ...exceptions import BentoMLException
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..configuration import is_editable_bentoml
from ..utils.pkg import source_locations

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

BENTOML_DEV_BUILD = "BENTOML_BUNDLE_LOCAL_BUILD"


def build_bentoml_sdist(
    target_path: str, *, _internal_protocol_version: str = LATEST_PROTOCOL_VERSION
) -> str | None:
    """
    This is for BentoML developers to create Bentos that contains the local bentoml
    build based on their development branch. To disable this behavior, one should
    set envar :code:`BENTOML_BUNDLE_LOCAL_BUILD=false` before building a Bento.

    Returns the sdist filename if the build is successful, otherwise None.
    """
    import tomli_w

    from ..configuration import BENTOML_VERSION

    if os.environ.get(BENTOML_DEV_BUILD, "true").lower() == "false":
        return

    if not is_editable_bentoml():
        # skip this entirely if BentoML is installed from PyPI
        return

    # Find bentoml module path
    # This will be $GIT_ROOT/src/bentoml
    module_location = source_locations("bentoml")
    if not module_location:
        raise BentoMLException("Could not find bentoml module location.")
    bentoml_path = Path(module_location)
    bentoml_project = bentoml_path.parent.parent

    if not (
        bentoml_path / "grpc" / _internal_protocol_version / "service_pb2.py"
    ).exists():
        raise ModuleNotFoundError(
            f"Generated stubs for version {_internal_protocol_version} are missing. Make sure to run '{bentoml_path.parent.as_posix()}/tools/generate-grpc-stubs {_internal_protocol_version}' beforehand to generate gRPC stubs."
        ) from None

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if not (bentoml_project / "pyproject.toml").exists():
        logger.warning(
            "Custom BentoML build is detected. For a Bento to use the same build at serving time, add your custom BentoML build to the pip packages list, e.g. `packages=['git+https://github.com/bentoml/bentoml.git@13dfb36']`"
        )
        return

    logger.debug(
        "BentoML is installed in `editable` mode; building BentoML distribution with the local BentoML code base. The built wheel file will be included in the target bento."
    )

    def exclude_pycache(tarinfo: tarfile.TarInfo):
        if "__pycache__" in tarinfo.name or tarinfo.name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    # Freeze the version in pyproject.toml
    with open(bentoml_project / "pyproject.toml", "rb") as f:
        pyproject_toml = tomllib.load(f)

    pyproject_toml["project"]["version"] = BENTOML_VERSION
    if (
        "dynamic" in pyproject_toml["project"]
        and "version" in pyproject_toml["project"]["dynamic"]
    ):
        pyproject_toml["project"]["dynamic"].remove("version")

    pyproject_io = BytesIO()
    tomli_w.dump(pyproject_toml, pyproject_io)

    # Make a tarball of the BentoML source code
    base_name = f"bentoml-{BENTOML_VERSION}"
    sdist_filename = f"{base_name}.tar.gz"
    files_to_include = ["src", "LICENSE", "README.md"]

    if not Path(target_path).exists():
        Path(target_path).mkdir(parents=True, exist_ok=True)

    with tarfile.open(Path(target_path, sdist_filename), "w:gz") as tar:
        for file in files_to_include:
            tar.add(
                bentoml_project / file,
                arcname=f"{base_name}/{file}",
                filter=exclude_pycache,
            )
        tarinfo = tar.gettarinfo(
            bentoml_project / "pyproject.toml", arcname=f"{base_name}/pyproject.toml"
        )
        tarinfo.size = pyproject_io.tell()
        pyproject_io.seek(0)
        tar.addfile(tarinfo, pyproject_io)

    return sdist_filename


def build_git_repo(url: str, ref: str, subdirectory: str | None, dst_path: str) -> str:
    """
    Download the git repo at the given url and ref, and build an sdist from it at dst_path
    """
    import shutil
    import subprocess

    from ...exceptions import BentoMLException
    from ..configuration import get_uv_command

    name = os.path.splitext(os.path.basename(url))[0]

    with fs.open_fs("temp://") as temp_fs:
        dest_dir = temp_fs.getsyspath(name)
        git_command = ["git", "clone", "--filter=blob:none", "--quiet", url, dest_dir]
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            subprocess.check_call(
                git_command,
                env=env,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            if ref:
                subprocess.check_call(
                    ["git", "fetch", "-q", url, ref],
                    cwd=dest_dir,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                )
                subprocess.check_call(
                    ["git", "checkout", "FETCH_HEAD"],
                    cwd=dest_dir,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                )
        except subprocess.CalledProcessError as e:
            raise BentoMLException(
                f"Failed to clone git repository {url}: {e.stderr}"
            ) from e
        source_dir = os.path.join(dest_dir, subdirectory) if subdirectory else dest_dir
        build_cmd = [*get_uv_command(), "build", "--sdist"]
        subprocess.check_call(build_cmd, cwd=source_dir)
        sdist = next(Path(source_dir).glob("dist/*.tar.gz"))
        logger.info(f"Built sdist {sdist.name}")
        os.makedirs(dst_path, exist_ok=True)
        shutil.move(sdist, dst_path)
        return sdist.name
