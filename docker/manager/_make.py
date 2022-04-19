from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from itertools import product

import fs
import attrs
import cattrs
import attrs.converters
from fs.base import FS
from simple_di import inject
from simple_di import Provide

from ._utils import send_log
from ._utils import inject_deepcopy_args
from ._utils import preprocess_template_paths
from ._configuration import DockerManagerContainer

if TYPE_CHECKING:

    GenericDict = t.Dict[str, t.Any]
    DictStrList = t.Dict[str, t.List[str]]

logger = logging.getLogger(__name__)


# CUDA-related
CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{suffixes}"
ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{suffixes}"
)

# our tag format
TAG_FORMAT = "{release_type}-python{python_version}-{suffixes}"


@attrs.define
class CUDALibraryVersion:
    version: str
    major_version: str = attrs.field(
        default=attrs.Factory(lambda self: self.version.split(".")[0], takes_self=True)
    )


@attrs.define
class CUDAVersion:
    major: str
    minor: str
    patch: str
    full: str


@attrs.define
class CUDAArchitecture:
    requires: str
    cudnn: t.Optional[CUDALibraryVersion]
    components: t.Dict[str, CUDALibraryVersion]


@attrs.define
class CUDA:
    version: CUDAVersion
    amd64: CUDAArchitecture
    arm64v8: CUDAArchitecture
    ppc64le: t.Optional[CUDAArchitecture]

    cuda_url_suffixes: str
    cuda_repo_url: str
    ml_repo_url: str


cattrs.register_structure_hook(CUDA, lambda value, _: cattrs.unstructure(value))


@inject_deepcopy_args
def generate_cuda_context(
    dependencies: t.Optional[t.Dict[str, GenericDict]]
) -> t.Union[CUDA, GenericDict]:

    if dependencies is None:
        return {}

    cuda_info = dependencies.pop("cuda")

    version = t.cast(str, cuda_info.pop("version"))
    major, minor, patch = version.split(".")

    cuda_url_suffixes = cuda_info.pop("cuda_url_suffixes")
    cuda_supported_arch: GenericDict = {
        "cuda_url_suffixes": cuda_url_suffixes,
        "version": CUDAVersion(major=major, minor=minor, patch=patch, full=version),
        "cuda_repo_url": CUDA_REPO_URL.format(suffixes=cuda_url_suffixes),
        "ml_repo_url": ML_REPO_URL.format(suffixes=cuda_url_suffixes),
    }

    for arch, requirements in cuda_info.items():
        if requirements is None:
            continue
        components = requirements["components"]
        requires = requirements["requires"]

        def get_cudnn_version() -> t.Optional[CUDALibraryVersion]:
            for key in components:
                if key.startswith("cudnn"):
                    cudnn = components.pop(key)
                    return CUDALibraryVersion(version=cudnn["version"])
            send_log(
                "current CUDA components doesn't support CUDNN.",
                _manager_level=logging.DEBUG,
            )
            return

        cuda_supported_arch[arch] = CUDAArchitecture(
            cudnn=get_cudnn_version(),
            components={
                k: CUDALibraryVersion(version=v["version"])
                for k, v in components.items()
            },
            requires=requires,
        )
    return cattrs.structure(cuda_supported_arch, CUDA)


def generate_envars(
    args: GenericDict, updater: t.Union[str, GenericDict, t.List[str]]
) -> GenericDict:
    # Args will have format ARG=foobar
    if isinstance(updater, str):
        arg, _, value = updater.partition("=")
        args[arg] = value
    elif isinstance(updater, list):
        for v in updater:
            generate_envars(args, v)
    elif isinstance(updater, dict):
        for arg, value in updater.items():
            args[arg] = value
    else:
        send_log(
            f"cannot add to args dict with unknown type {type(updater)}",
            _manager_level=logging.ERROR,
        )
    return args


@attrs.define
class DockerfileGenerationContext:

    distros: str
    release_type: str = attrs.field(
        validator=lambda _, __, value: set(value).issubset(
            DockerManagerContainer.RELEASE_TYPE_HIERARCHY
        ),
    )
    architecture: str = attrs.field(
        validator=lambda _, __, value: value
        in DockerManagerContainer.SUPPORTED_ARCHITECTURE_TYPE
    )

    conda: bool
    header: str
    suffixes: str
    base_image: str
    templates_dir: str
    envars: t.Dict[str, str]
    cuda: t.Optional[CUDA] = attrs.field(converter=generate_cuda_context)

    # provides a mapping for base and optionally conda base image tags
    base_tags: t.Dict[str, str] = attrs.field(init=False)

    # provides a mapping for templates/input_files to generated/output_files
    paths_mapping: t.Dict[str, str] = attrs.field(
        default=attrs.Factory(
            lambda self: self.generate_paths_mapping(), takes_self=True
        )
    )

    @inject
    def __attrs_post_init__(
        self,
        *,
        bentoml_version: str = Provide[DockerManagerContainer.bentoml_version],
    ):
        self.envars = generate_envars(
            {
                "BENTOML_VERSION": bentoml_version,
                "PYTHON_VERSION": "",
            },
            self.envars,
        )
        self.base_tags = generate_base_tags(suffixes=self.suffixes, conda=self.conda)

    @inject
    def generate_paths_mapping(
        self, *, _templates_fs: FS = Provide[DockerManagerContainer.templates_fs]
    ) -> t.Dict[str, str]:
        input_paths = [
            f
            for f in _templates_fs.walk.files(filter=[f"*{self.release_type}-*.j2"])
            if self.templates_dir in f
        ]
        return {
            inp: fs.path.join(
                f"/{DockerManagerContainer.docker_package}",
                self.distros,
                self.release_type,
                preprocess_template_paths(inp, self.architecture),
            )
            for inp in input_paths
        }


@attrs.define
class DistrosManifest:
    templates_dir: str
    suffixes: str
    base_image: str
    conda: bool
    ignore_python: t.Optional[t.List[str]]
    header: str
    envars: t.List[str]
    architectures: t.List[str]
    release_types: t.List[str]
    dependencies: t.Dict[str, GenericDict]

    python_versions: t.Iterable[str] = attrs.field(init=False)

    def __attrs_post_init__(self):
        if self.ignore_python is None:
            self.python_versions = DockerManagerContainer.SUPPORTED_PYTHON_VERSION
        else:
            self.python_versions = tuple(
                [
                    i
                    for i in DockerManagerContainer.SUPPORTED_PYTHON_VERSION
                    if i not in self.ignore_python
                ]
            )


@inject
def generate_base_tags(
    suffixes: str,
    conda: bool,
    *,
    docker_package: str = DockerManagerContainer.docker_package,
) -> t.Dict[str, str]:
    base_tag = {}
    base_ = (
        docker_package
        + ":"
        + TAG_FORMAT.format(
            release_type="base",
            python_version="$PYTHON_VERSION",
            suffixes=suffixes,
        )
    )
    base_tag = {"base": base_, "conda": ""}
    if conda:
        base_tag["conda"] = base_ + "-conda"
    return base_tag


@inject
def generate_releases_tags_mapping(
    distros_name: str,
    distros: DistrosManifest,
    *,
    bentoml_version: str = Provide[DockerManagerContainer.bentoml_version],
    skip_base_image: bool = True,
) -> t.List[t.Tuple[str, str]]:
    # distros is the dictionary representation from manifest file.
    metadata = []
    prefix_ = f"{DockerManagerContainer.docker_package}:"

    for rt, python_version in product(distros.release_types, distros.python_versions):
        if skip_base_image and rt == "base":
            continue

        git_tree_path = fs.path.join(
            f"/{DockerManagerContainer.docker_package}", distros_name, rt
        )

        if rt not in ["runtime", "cudnn"]:
            tag = TAG_FORMAT.format(
                release_type=rt,
                python_version=python_version,
                suffixes=distros.suffixes,
            )
        else:
            tag = "-".join(
                [
                    TAG_FORMAT.format(
                        release_type=bentoml_version,
                        python_version=python_version,
                        suffixes=distros.suffixes,
                    ),
                    rt,
                ]
            )
        release_tags = f"{prefix_}{tag}"
        metadata.append((release_tags, fs.path.combine(git_tree_path, "Dockerfile")))
        if distros.conda:
            metadata.append(
                (
                    release_tags + "-conda",
                    fs.path.combine(git_tree_path, "Dockerfile-conda"),
                )
            )
    return metadata
