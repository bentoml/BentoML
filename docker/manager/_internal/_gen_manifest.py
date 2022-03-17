from __future__ import annotations

import os
import typing as t
import itertools
from uuid import uuid1
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial

import fs
import yaml
from yamlinclude import YamlIncludeConstructor

from .utils import send_log
from .utils import render_template
from .utils import DOCKERFILE_BUILD_HIERARCHY
from .utils import SUPPORTED_ARCHITECTURE_TYPE

MANIFEST_FILENAME = "{}.cuda_v{}.yaml"

DOCKER_DIRECTORY = Path(os.path.dirname(__file__)).parent.parent

if TYPE_CHECKING:
    from fs.base import FS

    IncludeMapping = t.Dict[str, str]

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader,
    base_dir=DOCKER_DIRECTORY.joinpath("manager").__fspath__(),
)


CUDA_ARCHITECTURE_PER_DISTRO = {
    "debian11": ["amd64", "arm64v8"],
    "debian10": ["amd64", "arm64v8"],
    "ubi8": ["amd64", "arm64v8", "ppc64le"],
    "ubi7": ["amd64", "arm64v8", "ppc64le"],
}

ARCHITECTURE_PER_DISTRO = {
    "alpine3.14": SUPPORTED_ARCHITECTURE_TYPE,
    "debian11": SUPPORTED_ARCHITECTURE_TYPE,
    "debian10": SUPPORTED_ARCHITECTURE_TYPE,
    "ubi8": SUPPORTED_ARCHITECTURE_TYPE,
    "ubi7": ["amd64", "s390x", "ppc64le"],
    "amazonlinux2": ["amd64", "arm64v8"],
}


def walk_include_dir(
    include_fs: FS,
    templates_dir: str,
    checker: t.Iterable[t.Any],
    filter_key: t.Callable[[str], t.Any] = lambda x: x,
) -> IncludeMapping:
    results = {}

    templates_fs = include_fs.makedirs(templates_dir, recreate=True)
    for check, p in itertools.product(
        checker, templates_fs.walk.files(filter=["*.yaml"])
    ):
        if check in p:
            p = p.strip("/").strip(".yaml")
            results[filter_key(p)] = f"!include include.d/{templates_dir}/{p}.yaml"
    return results


def unpack_include_item(include_: IncludeMapping, indent: int = 0) -> str:
    mem_fs = fs.open_fs("mem://")
    tmp_file = partial(mem_fs.open, f".{uuid1()}.yml", encoding="utf-8")
    include_string = "\n".join([f"{k}: {v}\n" for k, v in include_.items()])

    with tmp_file("w") as inf:
        yaml.dump(
            yaml.load(include_string.encode("utf-8"), Loader=yaml.FullLoader), inf
        )
    with tmp_file("r") as res:
        indentation = indent * " "
        res = indentation.join(res.readlines())

    mem_fs.close()

    return res


def gen_manifest(
    docker_package: str,
    cuda_version: str,
    supported_distro: t.Iterable[str],
    *,
    overwrite: bool,
    docker_fs: FS,
    registries: t.Iterable[str],
):
    name = MANIFEST_FILENAME.format(docker_package, cuda_version)

    include_path = fs.path.join("manager", "include.d")
    include_fs = docker_fs.makedirs(include_path, recreate=True)
    manifest_fs = docker_fs.opendir("manifest")

    cuda_map = walk_include_dir(
        include_fs, "cuda", [cuda_version], filter_key=lambda x: x.split(".")[-1]
    )
    reg_map = walk_include_dir(include_fs, "registry", registries)

    reg = unpack_include_item(reg_map)

    if not manifest_fs.exists(name) or overwrite:

        spec_tmpl = {
            "cuda_version": cuda_version,
            "architectures": SUPPORTED_ARCHITECTURE_TYPE,
            "release_types": DOCKERFILE_BUILD_HIERARCHY,
            "cuda_architecture_per_distro": CUDA_ARCHITECTURE_PER_DISTRO,
            "cuda_mapping": cuda_map,
            "registries": reg,
            "supported_distros": supported_distro,
            "architecture_per_distros": ARCHITECTURE_PER_DISTRO,
            "templates_entries": {
                "debian11": "debian",
                "debian10": "debian",
                "ubi8": "rhel",
                "ubi7": "rhel",
                "amazonlinux2": "ami2",
                "alpine3.14": "alpine",
            },
        }

        render_template(
            "spec.yaml.j2",
            include_fs,
            "/",
            manifest_fs,
            output_name=name,
            overwrite_output_path=overwrite,
            preserve_output_path_name=True,
            create_as_dir=False,
            custom_function={"unpack_include": unpack_include_item},
            **spec_tmpl,
        )
    else:
        if not overwrite:
            send_log(
                f"{manifest_fs.getsyspath(name)} won't be overwritten."
                " To overwrite pass `--overwrite`",
                extra={"markup": True},
            )
            return
