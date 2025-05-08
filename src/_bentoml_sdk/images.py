from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
import typing as t
from pathlib import Path

import attrs
import fs

from bentoml._internal.bento.bento import ImageInfo
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration import DEFAULT_LOCK_PLATFORM
from bentoml._internal.configuration import get_bentoml_requirement
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration import get_quiet_mode
from bentoml._internal.container.frontend.dockerfile import CONTAINER_METADATA
from bentoml._internal.container.frontend.dockerfile import CONTAINER_SUPPORTED_DISTROS
from bentoml.exceptions import BentoMLConfigException
from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:
    from fs.base import FS

    from bentoml._internal.bento.build_config import BentoEnvSchema

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger("bentoml.build")

DEFAULT_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


@attrs.define
class Image:
    """A class defining the environment requirements for bento."""

    base_image: str = ""
    distro: str = "debian"
    python_version: str = DEFAULT_PYTHON_VERSION
    commands: t.List[str] = attrs.field(factory=list)
    lock_python_packages: bool = True
    pack_git_packages: bool = True
    python_requirements: str = ""
    post_commands: t.List[str] = attrs.field(factory=list)
    scripts: t.Dict[str, str] = attrs.field(factory=dict, init=False)
    _after_pip_install: bool = attrs.field(init=False, default=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if not self.base_image:
            if self.distro not in CONTAINER_METADATA:
                raise BentoMLConfigException(
                    f"Unsupported distro: {self.distro}, expected one of {CONTAINER_SUPPORTED_DISTROS}"
                )
            python_version = self.python_version
            if self.distro in ("ubi8",):
                python_version = python_version.replace(".", "")
            distro_config = CONTAINER_METADATA[self.distro]
            self.base_image = distro_config["python"]["image"].format(
                spec_version=python_version
            )
        if self.distro in CONTAINER_METADATA:
            self.commands.insert(
                0, CONTAINER_METADATA[self.distro]["default_install_command"]
            )

    def system_packages(self, *packages: str) -> t.Self:
        if self.distro not in CONTAINER_METADATA:
            raise BentoMLConfigException(
                f"Unsupported distro: {self.distro}, expected one of {CONTAINER_SUPPORTED_DISTROS}"
            )
        logger.info(
            "Adding system packages using distro %s's package manager", self.distro
        )
        self.commands.append(
            CONTAINER_METADATA[self.distro]["install_command"].format(
                packages=" ".join(packages)
            )
        )
        return self

    def requirements_file(self, file_path: str) -> t.Self:
        """Add a requirements file to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").requirements_file("requirements.txt")
        """
        if self.post_commands:
            raise BentoMLConfigException("Can't separate adding python requirements")
        self.python_requirements += Path(file_path).read_text().rstrip("\n") + "\n"
        self._after_pip_install = True
        return self

    def pyproject_toml(self, file_path: str) -> t.Self:
        """Add a pyproject.toml to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").pyproject_toml("pyproject.toml")
        """
        if self.post_commands:
            raise BentoMLConfigException("Can't separate adding python requirements")
        with Path(file_path).open("rb") as f:
            pyproject_toml = tomllib.load(f)
        dependencies = pyproject_toml.get("project", {}).get("dependencies", {})
        self.python_packages(*dependencies)
        return self

    def python_packages(self, *packages: str) -> t.Self:
        """Add python dependencies to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest")\
                .python_packages("numpy", "pandas")\
                .requirements_file("requirements.txt")
        """
        if self.post_commands:
            raise BentoMLConfigException("Can't separate adding python requirements")
        if not packages:
            raise BentoMLConfigException("No packages provided")
        self.python_requirements += "\n".join(packages) + "\n"
        self._after_pip_install = True
        return self

    def run(self, command: str) -> t.Self:
        """Add a command to the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").run("echo 'Hello, World!'")
        """
        commands = self.post_commands if self._after_pip_install else self.commands
        commands.append(command)
        return self

    def run_script(self, script: str) -> t.Self:
        """Run a script in the image. Supports chaining call.

        Example:

        .. code-block:: python

            image = Image("debian:latest").run_script("script.sh")
        """
        commands = self.post_commands if self._after_pip_install else self.commands
        script = Path(script).resolve().as_posix()
        # Files under /env/docker will be copied into the env image layer
        target_script = (
            f"./env/docker/script__{hashlib.md5(script.encode()).hexdigest()}"
        )
        commands.append(f"chmod +x {target_script} && {target_script}")
        self.scripts[script] = target_script
        return self

    def freeze(
        self, bento_fs: FS, envs: list[BentoEnvSchema], platform_: str | None = None
    ) -> ImageInfo:
        """Freeze the image to an ImageInfo object for build."""
        python_requirements = self._freeze_python_requirements(bento_fs, platform_)
        from importlib import resources

        from _bentoml_impl.docker import generate_dockerfile
        from bentoml._internal.utils.filesystem import copy_file_to_fs_folder

        # Prepare env/python files
        py_folder = fs.path.join("env", "python")
        bento_fs.makedirs(py_folder, recreate=True)
        reqs_txt = fs.path.join(py_folder, "requirements.txt")
        bento_fs.writetext(reqs_txt, python_requirements)
        info = ImageInfo(
            base_image=self.base_image,
            python_version=self.python_version,
            commands=self.commands,
            python_requirements=python_requirements,
            post_commands=self.post_commands,
        )
        # Prepare env/docker files
        docker_folder = fs.path.join("env", "docker")
        bento_fs.makedirs(docker_folder, recreate=True)
        dockerfile_path = fs.path.join(docker_folder, "Dockerfile")
        bento_fs.writetext(
            dockerfile_path,
            generate_dockerfile(info, bento_fs, enable_buildkit=False, envs=envs),
        )
        for script_name, target_path in self.scripts.items():
            copy_file_to_fs_folder(script_name, bento_fs, dst_filename=target_path)

        with resources.path(
            "bentoml._internal.container.frontend.dockerfile", "entrypoint.sh"
        ) as entrypoint_path:
            copy_file_to_fs_folder(str(entrypoint_path), bento_fs, docker_folder)
        return info

    def _freeze_python_requirements(
        self, bento_fs: FS, platform_: str | None = None
    ) -> str:
        from pip_requirements_parser import RequirementsFile

        from bentoml._internal.bento.bentoml_builder import build_bentoml_sdist
        from bentoml._internal.bento.build_config import PythonOptions
        from bentoml._internal.configuration import get_uv_command

        py_folder = fs.path.join("env", "python")
        bento_fs.makedirs(py_folder, recreate=True)
        requirements_in = Path(
            bento_fs.getsyspath(fs.path.join(py_folder, "requirements.in"))
        )
        requirements_in.write_text(self.python_requirements)
        py_req = fs.path.join("env", "python", "requirements.txt")
        requirements_out = Path(bento_fs.getsyspath(py_req))
        # XXX: RequirementsFile.from_string() does not work due to bugs
        requirements_file = RequirementsFile.from_file(str(requirements_in))
        has_bentoml_req = any(
            req.name and req.name.lower() == "bentoml" and req.link is not None
            for req in requirements_file.requirements
        )
        src_wheels = fs.path.join("src", "wheels")
        wheels_folder = fs.path.join("env", "python", "wheels")
        if bento_fs.exists(src_wheels):
            bento_fs.movedir(src_wheels, wheels_folder, create=True)
        with requirements_in.open("w") as f:
            f.write(requirements_file.dumps(preserve_one_empty_line=True))
            if not has_bentoml_req:
                sdist_name = build_bentoml_sdist(bento_fs.getsyspath(wheels_folder))
                bento_req = get_bentoml_requirement()
                if bento_req is not None:
                    logger.info(
                        "Adding BentoML requirement to the image: %s.", bento_req
                    )
                    f.write(f"{bento_req}\n")
                elif sdist_name is not None:
                    f.write(f"./wheels/{sdist_name}\n")
        if not self.lock_python_packages:
            requirements_out.parent.mkdir(parents=True, exist_ok=True)
            requirements_out.write_text(requirements_in.read_text())
            PythonOptions.fix_dep_urls(
                str(requirements_out),
                bento_fs.getsyspath(wheels_folder),
                self.pack_git_packages,
            )
            return requirements_out.read_text()
        lock_args = [
            str(requirements_in),
            "--no-header",
            f"--output-file={requirements_out}",
            "--emit-index-url",
            "--emit-find-links",
            "--no-annotate",
        ]
        if get_debug_mode():
            lock_args.append("--verbose")
        else:
            lock_args.append("--quiet")
        logger.info("Locking PyPI package versions.")
        if platform_:
            lock_args.extend(["--python-platform", platform_])
        elif platform.system() != "Linux" or platform.machine() != "x86_64":
            logger.info(
                "Locking packages for %s. Pass `--platform` option to specify the platform.",
                DEFAULT_LOCK_PLATFORM,
            )
            lock_args.extend(["--python-platform", DEFAULT_LOCK_PLATFORM])
        cmd = [*get_uv_command(), "pip", "compile", *lock_args]
        try:
            subprocess.check_call(
                cmd,
                text=True,
                stderr=subprocess.DEVNULL if get_quiet_mode() else None,
                cwd=bento_fs.getsyspath(py_folder),
            )
        except subprocess.CalledProcessError as e:
            raise BentoMLException(
                "Failed to lock PyPI packages. Add `--debug` option to see more details.\n"
                "You see this error because you set `lock_packages=true` in the image config.\n"
                "Learn more at https://docs.bentoml.com/en/latest/reference/bentoml/bento-build-options.html#pypi-package-locking"
            ) from e

        PythonOptions.fix_dep_urls(
            str(requirements_out),
            bento_fs.getsyspath(wheels_folder),
            self.pack_git_packages,
        )
        return requirements_out.read_text()


def populate_image_from_build_config(
    image: Image | None, build_config: BentoBuildConfig, build_ctx: str
) -> Image | None:
    from bentoml._internal.utils.filesystem import resolve_user_filepath

    fallback_message = "fallback to bento v1" if image is None else "it will be ignored"
    if not build_config.conda.is_empty():
        logger.warning(
            "conda options are not supported by bento v2, %s", fallback_message
        )
        if image is None:
            return None
    docker_options = build_config.docker
    if docker_options.cuda_version is not None:
        logger.warning(
            "docker.cuda_version is not supported by bento v2, %s", fallback_message
        )
        if image is None:
            return None
    if docker_options.dockerfile_template is not None:
        logger.warning(
            "docker.dockerfile_template is not supported by bento v2, %s",
            fallback_message,
        )
        if image is None:
            return None

    image_params = {}
    if docker_options.base_image is not None:
        image_params["base_image"] = docker_options.base_image
    if docker_options.distro is not None:
        image_params["distro"] = docker_options.distro
    if docker_options.python_version is not None:
        image_params["python_version"] = docker_options.python_version

    python_options = build_config.python
    if python_options.wheels:
        logger.warning(
            "python.wheels is not supported by bento v2, %s\nAdd the wheels directly into your project instead.",
            fallback_message,
        )
        if image is None:
            return None
    if python_options.lock_packages is not None:
        image_params["lock_python_packages"] = python_options.lock_packages
    if python_options.pack_git_packages is not None:
        image_params["pack_git_packages"] = python_options.pack_git_packages
    image = (
        Image(**image_params) if image is None else attrs.evolve(image, **image_params)
    )
    if docker_options.system_packages:
        image.system_packages(*docker_options.system_packages)
    if python_options.index_url:
        image.python_packages(f"--index-url {python_options.index_url}")
    if python_options.no_index:
        image.python_packages("--no-index")
    if python_options.trusted_host:
        image.python_packages(
            *(f"--trusted-host {h}" for h in python_options.trusted_host)
        )
    if python_options.extra_index_url:
        image.python_packages(
            *(f"--extra-index-url {url}" for url in python_options.extra_index_url)
        )
    if python_options.find_links:
        image.python_packages(
            *(f"--find-links {link}" for link in python_options.find_links)
        )
    if python_options.requirements_txt:
        image.requirements_file(
            resolve_user_filepath(python_options.requirements_txt, build_ctx)
        )
    if python_options.packages:
        image.python_packages(*python_options.packages)
    if docker_options.setup_script:
        image.run_script(docker_options.setup_script)
    return image
