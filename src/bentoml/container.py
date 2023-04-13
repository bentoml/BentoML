"""
User facing python APIs for building a OCI-complicant image.
"""

from __future__ import annotations

import os
import sys
import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from .exceptions import BentoMLException
from ._internal.container import build as _internal_build
from ._internal.container import health
from ._internal.container import get_backend
from ._internal.container import register_backend
from ._internal.container import (
    construct_containerfile as _internal_construct_containerfile,
)
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.tag import Tag
    from ._internal.bento import BentoStore
    from ._internal.types import PathType
    from ._internal.container.base import ArgType

logger = logging.getLogger(__name__)


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["docker"] = ...,
    image_tag: tuple[str, ...] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    file: PathType | None = ...,
    context_path: PathType | None = ...,
    add_host: dict[str, str] | ArgType = ...,
    build_arg: dict[str, str] | ArgType = ...,
    cache_from: str | dict[str, str] | ArgType = ...,
    disable_content_trust: t.Literal[True, False] = ...,
    iidfile: PathType | None = ...,
    isolation: t.Literal["default", "process", "hyperv"] | None = ...,
    label: dict[str, str] | ArgType = ...,
    network: str | None = ...,
    no_cache: t.Literal[True, False] = ...,
    output: str | dict[str, str] | ArgType = ...,
    platform: str | ArgType = ...,
    progress: t.Literal["auto", "tty", "plain"] = ...,
    pull: t.Literal[True, False] = ...,
    quiet: t.Literal[True, False] = ...,
    secret: str | dict[str, str] | ArgType = ...,
    ssh: str | ArgType = ...,
    target: str | ArgType = ...,
):
    ...


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["buildctl"] = ...,
    image_tag: tuple[str] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    file: PathType | None = ...,
    context_path: PathType | None = ...,
    output: str | dict[str, str] | ArgType = ...,
    progress: t.Literal["auto", "tty", "plain"] | ArgType = ...,
    trace: PathType | None = ...,
    local: dict[str, str] | ArgType = ...,
    frontend: str | None = ...,
    no_cache: t.Literal[True, False] = ...,
    export_cache: str | dict[str, str] | ArgType = ...,
    import_cache: str | dict[str, str] | ArgType = ...,
    secret: str | dict[str, str] | ArgType = ...,
    allow: str | ArgType = ...,
    ssh: str | ArgType = ...,
    metadata_file: PathType | None = ...,
    opt: tuple[str, ...] | dict[str, str | tuple[str, ...]] | None = ...,
):
    ...


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["buildx"] = ...,
    image_tag: tuple[str] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    file: PathType | None = ...,
    tag: tuple[str] | None = ...,
    context_path: PathType = ...,
    add_host: dict[str, str] | ArgType = ...,
    allow: str | ArgType = ...,
    build_arg: dict[str, str] | ArgType = ...,
    build_context: dict[str, str] | ArgType = ...,
    builder: str | None = ...,
    cache_from: str | dict[str, str] | ArgType = ...,
    cache_to: str | dict[str, str] | ArgType = ...,
    cgroup_parent: PathType | None = ...,
    iidfile: PathType | None = ...,
    label: dict[str, str] | ArgType = ...,
    load: bool = True,
    metadata_file: PathType | None = ...,
    network: str | None = ...,
    no_cache: t.Literal[True, False] = ...,
    no_cache_filter: str | dict[str, str] | ArgType = ...,
    output: str | dict[str, str] | ArgType = ...,
    platform: str | ArgType = ...,
    progress: t.Literal["auto", "tty", "plain"] = "auto",
    pull: t.Literal[True, False] = ...,
    push: t.Literal[True, False] = ...,
    quiet: t.Literal[True, False] = ...,
    secret: str | dict[str, str] | ArgType = ...,
    shm_size: int | None = ...,
    ssh: str | ArgType = ...,
    target: str | None = ...,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = ...,
):
    ...


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["nerdctl"] = ...,
    image_tag: tuple[str] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    file: PathType | None = ...,
    tag: tuple[str] | None = ...,
    context_path: PathType = ".",
    build_arg: dict[str, str] | ArgType = ...,
    buildkit_host: str | None = ...,
    cache_from: str | dict[str, str] | ArgType = ...,
    cache_to: str | dict[str, str] | ArgType = ...,
    iidfile: PathType | None = ...,
    ipfs: t.Literal[True, False] = ...,
    label: dict[str, str] | ArgType = ...,
    no_cache: t.Literal[True, False] = ...,
    output: str | dict[str, str] | ArgType = ...,
    platform: str | ArgType = ...,
    progress: t.Literal["auto", "tty", "plain"] | ArgType = ...,
    quiet: t.Literal[True, False] = ...,
    rm: t.Literal[True, False] = ...,
    secret: str | dict[str, str] | ArgType = ...,
    ssh: str | ArgType = ...,
    target: str | None = ...,
    # global flags
    address: str | None = ...,
    host: str | None = ...,
    cgroup_manager: str | None = ...,
    cni_netconfpath: PathType | None = ...,
    cni_path: PathType | None = ...,
    data_root: PathType | None = ...,
    debug: t.Literal[True, False] = ...,
    debug_full: t.Literal[True, False] = ...,
    hosts_dir: str | ArgType = ...,
    insecure_registry: t.Literal[True, False] = ...,
    namespace: str | None = ...,
    snapshotter: str | None = ...,
    storage_driver: str | None = ...,
):
    ...


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["podman"] = ...,
    image_tag: tuple[str] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    file: PathType | None = ...,
    context_path: PathType | None = ...,
    add_host: dict[str, str] | ArgType = ...,
    all_platforms: t.Literal[True, False] = ...,
    annotation: dict[str, str] | ArgType = ...,
    label: dict[str, str] | ArgType = ...,
    arch: str | None = ...,
    authfile: PathType | None = ...,
    build_arg: dict[str, str] | ArgType = ...,
    build_context: dict[str, str] | ArgType = ...,
    cache_from: str | None = ...,
    cache_to: str | None = ...,
    cache_ttl: str | None = ...,
    cap_add: str | ArgType = ...,
    cap_drop: str | ArgType = ...,
    cert_dir: PathType | None = ...,
    cgroup_parent: PathType | None = ...,
    cgroupns: str | None = ...,
    cpp_flag: ArgType = ...,
    cpu_period: int | None = ...,
    cpu_quota: int | None = ...,
    cpu_shares: int | None = ...,
    cpuset_cpus: str | None = ...,
    cpuset_mems: str | None = ...,
    creds: str | dict[str, str] | ArgType = ...,
    decryption_key: str | dict[str, str] | ArgType = ...,
    device: str | ArgType = ...,
    disable_compression: t.Literal[True, False] = ...,
    dns: str | None = ...,
    dns_option: str | ArgType = ...,
    dns_search: str | ArgType = ...,
    env: str | dict[str, str] | ArgType = ...,
    force_rm: t.Literal[True, False] = ...,
    format: str | t.Literal["docker", "oci"] | None = ...,
    hooks_dir: str | ArgType = ...,
    http_proxy: t.Literal[True, False] = ...,
    identity_label: t.Literal[True, False] = ...,
    ignorefile: PathType | None = ...,
    iidfile: PathType | None = ...,
    ipc: str | PathType | None = ...,
    isolation: str | None = ...,
    jobs: int | None = ...,
    layers: t.Literal[True, False] = ...,
    logfile: PathType | None = ...,
    manifest: str | None = ...,
    memory: str | None = ...,
    memory_swap: str | None = ...,
    network: str | None = ...,
    no_cache: t.Literal[True, False] = ...,
    no_hosts: t.Literal[True, False] = ...,
    omit_history: t.Literal[True, False] = ...,
    os: str | None = ...,
    os_feature: str | None = ...,
    os_version: str | None = ...,
    pid: PathType | None = ...,
    platform: str | ArgType = ...,
    output: str | dict[str, str] | ArgType = ...,
    pull: t.Literal[
        True, False, "always", "true", "missing", "never", "false", "newer"
    ] = False,
    quiet: t.Literal[True, False] = ...,
    rm: t.Literal[True, False] = ...,
    retry: int | None = ...,
    retry_delay: int | None = ...,
    runtime: PathType | None = ...,
    runtime_flag: str | dict[str, str] | ArgType = ...,
    secret: str | dict[str, str] | ArgType = ...,
    security_opt: str | ArgType = ...,
    shm_size: str | None = ...,
    sign_by: str | None = ...,
    skip_unused_stages: t.Literal[True, False] = ...,
    squash: t.Literal[True, False] = ...,
    squash_all: t.Literal[True, False] = ...,
    ssh: str | ArgType = ...,
    stdin: t.Literal[True, False] = ...,
    target: str | None = ...,
    timestamp: int | None = ...,
    tls_verify: t.Literal[True, False] = ...,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = ...,
    unsetenv: str | ArgType = ...,
    userns: str | None = ...,
    userns_gid_map: str | tuple[str, str, str] | None = ...,
    userns_gid_map_group: str | None = ...,
    userns_uid_map: str | tuple[str, str, str] | None = ...,
    userns_uid_map_user: str | None = ...,
    uts: str | None = ...,
    variant: str | None = ...,
    volume: str | tuple[str, str, str] | None = ...,
):
    ...


@t.overload
def build(
    bento_tag: Tag | str,
    backend: t.Literal["buildah"] = ...,
    image_tag: tuple[str] | None = ...,
    features: t.Sequence[str] | None = ...,
    *,
    context_path: PathType = ...,
    file: PathType | None = ...,
    tag: tuple[str] | None = ...,
    add_host: dict[str, str] | ArgType = ...,
    annotation: dict[str, str] | ArgType = ...,
    label: dict[str, str] | ArgType = ...,
    arch: str | None = ...,
    authfile: PathType | None = ...,
    build_arg: dict[str, str] | ArgType = ...,
    cache_from: str | None = ...,
    cap_add: str | ArgType = ...,
    cap_drop: str | ArgType = ...,
    cert_dir: PathType | None = ...,
    cgroup_parent: PathType | None = ...,
    cni_config_dir: PathType | None = ...,
    cni_plugin_path: PathType | None = ...,
    compress: t.Literal[True, False] = ...,
    cpu_period: int | None = ...,
    cpu_quota: int | None = ...,
    cpu_shares: int | None = ...,
    cpuset_cpus: str | None = ...,
    cpuset_mems: str | None = ...,
    creds: str | dict[str, str] | ArgType = ...,
    decryption_key: str | dict[str, str] | ArgType = ...,
    device: str | ArgType = ...,
    disable_compression: t.Literal[True, False] = ...,
    dns: str | None = ...,
    dns_option: str | ArgType = ...,
    dns_search: str | ArgType = ...,
    force_rm: t.Literal[True, False] = ...,
    format: str | t.Literal["docker", "oci"] | None = ...,
    http_proxy: t.Literal[True, False] = ...,
    ignorefile: PathType | None = ...,
    iidfile: PathType | None = ...,
    ipc: str | PathType | None = ...,
    isolation: str | None = ...,
    jobs: int | None = ...,
    layers: t.Literal[True, False] = ...,
    logfile: PathType | None = ...,
    manifest: str | None = ...,
    memory: str | None = ...,
    memory_swap: str | None = ...,
    network: str | None = ...,
    no_cache: t.Literal[True, False] = ...,
    os: str | None = ...,
    pid: PathType | None = ...,
    platform: str | ArgType = ...,
    pull: t.Literal[True, False] = ...,
    pull_always: t.Literal[True, False] = ...,
    pull_never: t.Literal[True, False] = ...,
    quiet: t.Literal[True, False] = ...,
    rm: t.Literal[True, False] = ...,
    runtime: PathType | None = ...,
    runtime_flag: str | dict[str, str] | ArgType = ...,
    secret: str | dict[str, str] | ArgType = ...,
    security_opt: str | ArgType = ...,
    shm_size: str | None = ...,
    sign_by: str | None = ...,
    squash: t.Literal[True, False] = ...,
    ssh: str | ArgType = ...,
    stdin: t.Literal[True, False] = ...,
    target: str | None = ...,
    timestamp: int | None = ...,
    tls_verify: t.Literal[True, False] = ...,
    ulimit: str | dict[str, tuple[int, int]] | ArgType = ...,
    userns: str | None = ...,
    userns_gid_map_group: str | None = ...,
    userns_uid_map_user: str | None = ...,
    uts: str | None = ...,
    variant: str | None = ...,
    volume: str | tuple[str, str, str] | None = ...,
):
    ...


@inject
def build(
    bento_tag: Tag | str,
    backend: str = "docker",
    image_tag: tuple[str] | None = None,
    features: t.Sequence[str] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    **kwargs: t.Any,
):
    """
    Build any given BentoML into a OCI-compliant image.

    .. code-block:: python

        import bentoml

        bento = bentoml.get("pytorch_vgg:latest")
        bentoml.container.build(bento, backend='podman', features=["grpc", "tracing"])

    Args:
        bento_tag: Bento tag in format of ``NAME:VERSION``
        backend: The backend to use for building the image. Current supported builder backends
                 include ``docker``, ``podman``, ``buildah``, ``nerdctl``, ``buildctl``, and ``buildx``.

                 .. note::

                     ``buildx`` is a syntatic sugar for ``docker buildx build``. See https://docs.docker.com/build/.
                     The reason for this is that ``buildx`` used to be the default behaviour of ``bentoml containerize``.
        image_tag: Optional additional image tag to apply to the built image.
        features: Optional features to include in the container file. See :ref:`concepts/bento:Python Packages`
                  for additional BentoML features.
        **kwargs: Additional keyword arguments to pass to the builder backend. Refer to the above overload
                  for each of the supported arguments per backend.
    """
    from ._internal.container import determine_container_tag

    # Run healthcheck
    if not health(backend):
        raise BentoMLException("Failed to use backend %s." % backend)

    if "tag" not in kwargs:
        kwargs["tag"] = determine_container_tag(bento_tag, image_tag=image_tag)
    bento = _bento_store.get(bento_tag)

    logger.info("Building OCI-compliant image for %s with %s\n", bento.tag, backend)
    return _internal_build(bento_tag, backend, features=features, **kwargs)


@inject
def get_containerfile(
    bento_tag: Tag | str,
    output_path: str | None = None,
    enable_buildkit: bool = True,
    features: t.Sequence[str] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
):
    """
    Returns the generated container file for a given Bento.

    Note that the container file (Dockerfile) inside the Bento is minimal, whereas
    this utility functions returns the container file that :ref:`bentoml containerize <reference/cli:containerize>` will
    be using.

    .. note::

        If ``output_path`` is not specified, then the contents of the container file
        will be printed out to ``sys.stderr``. If provided, then the final container file
        will be written to that given path.

    Args:
        bento_tag: Given tag for the bento.
        output_path: Optional output path to write the final container file to.
                     Note that if ``output_path`` is a directory, then the targeted file
                     will be ``output_path + os.sep + "<bento_tag>.dockerfile"``.
        enable_buildkit: Whether the container file contains BuildKit syntax.
        features: Optional features to include in the container file. See :ref:`concepts/bento:Python Packages`
                  for additional BentoML features.
    """
    bento = _bento_store.get(bento_tag)
    with _internal_construct_containerfile(
        bento,
        enable_buildkit=enable_buildkit,
        features=features,
        add_header=True,
    ) as (_, final_containerfile):
        if output_path is not None:
            if os.path.isdir(output_path):
                output_path = output_path = os.path.join(
                    output_path, f"{bento.tag.path().replace(os.sep,'_')}.dockerfile"
                )
            logger.info(
                "Writting Containerfile for '%s' to '%s'.", bento.tag, output_path
            )
            shutil.copyfile(final_containerfile, output_path)
            return
        # otherwise we will just write this to stderr.
        with open(final_containerfile, "r") as f:
            sys.stderr.write(f.read())


__all__ = ["build", "health", "register_backend", "get_backend", "get_containerfile"]
