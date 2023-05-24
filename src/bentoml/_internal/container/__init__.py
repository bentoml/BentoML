from __future__ import annotations

import os
import sys
import types
import typing as t
import logging
import importlib
import contextlib
from typing import TYPE_CHECKING

import fs
import fs.mirror
from simple_di import inject
from simple_di import Provide

from .base import OCIBuilder
from ..utils import bentoml_cattr
from .generate import generate_containerfile
from ...exceptions import InvalidArgument
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..tag import Tag
    from .base import Arguments
    from ..bento import Bento
    from ..bento import BentoStore

    P = t.ParamSpec("P")

    class DefaultBackendImpl(types.ModuleType):
        BUILD_CMD: list[str] | None
        ENV: dict[str, str] | None
        BUILDKIT_SUPPORT: bool

        def find_binary(self) -> str | None:
            ...

        def construct_build_args(
            self, **kwargs: t.Any  # pylint: disable=unused-argument
        ) -> Arguments:
            ...

        def health(self) -> bool:
            ...

    DefaultBuilder: t.TypeAlias = t.Literal[
        "docker", "podman", "buildah", "buildx", "nerdctl", "buildctl"
    ]


logger = logging.getLogger(__name__)

BUILDER_REGISTRY: dict[str, OCIBuilder] = {}


DEFAULT_BACKENDS = frozenset(
    {"docker", "buildx", "buildah", "podman", "nerdctl", "buildctl"}
)


def register_default_backends():
    for backend in DEFAULT_BACKENDS:
        try:
            module = t.cast(
                "DefaultBackendImpl", importlib.import_module(f".{backend}", __name__)
            )
            register_backend(
                backend,
                health=module.health,
                binary=module.find_binary(),
                construct_build_args=module.construct_build_args,
                buildkit_support=module.BUILDKIT_SUPPORT,
                env=getattr(module, "ENV", None),
                build_cmd=getattr(module, "BUILD_CMD", None),
            )
        except NotImplementedError:
            logger.debug("%s is not yet implemented.", backend)
        except Exception as err:  # pylint: disable=broad-except
            raise ValueError(f"Failed to register backend '{backend}: {err}'") from None


@inject
def determine_container_tag(
    bento_tag: Tag | str,
    image_tag: tuple[str] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
):
    # NOTE: for tags strategy, we will always generate a default tag from the bento:tag
    # If '-t/--image-tag' is provided, we will use this tag provided by user.
    bento = _bento_store.get(bento_tag)
    tag = (str(bento.tag),)
    if image_tag is not None:
        assert isinstance(image_tag, tuple)
        tag = image_tag
    return tag


def enable_buildkit(
    *, backend: str | None = None, builder: OCIBuilder | None = None
) -> bool:
    # We will look for DOCKER_BUILDKIT in the environment variable.
    # as this will be our entry point to enable BuildKit.
    if "DOCKER_BUILDKIT" in os.environ:
        return bool(int(os.environ["DOCKER_BUILDKIT"]))
    # If the variable is not set, fallback to the default value
    # provided by the builder.
    if builder is not None:
        return builder.enable_buildkit
    elif backend is not None:
        return get_backend(backend).enable_buildkit
    else:
        raise ValueError("Either backend or builder must be provided.")


# XXX: Sync with BentoML extra dependencies found in pyproject.toml
FEATURES = frozenset(
    {
        "tracing",
        "grpc",
        "grpc-reflection",
        "grpc-channelz",
        "monitor-otlp",
        "triton",
        "aws",
        "all",
        "io",
        "io-file",
        "io-image",
        "io-pandas",
        "io-json",
        "tracing-zipkin",
        "tracing-jaeger",
        "tracing-otlp",
    }
)


@contextlib.contextmanager
def construct_containerfile(
    bento: Bento,
    enable_buildkit: bool = True,
    *,
    features: t.Sequence[str] | None = None,
    add_header: bool = False,
) -> t.Generator[tuple[str, str], None, None]:
    from ..bento.bento import BentoInfo
    from ..bento.build_config import DockerOptions

    dockerfile_path = "env/docker/Dockerfile"
    instruction: list[str] = []

    with fs.open_fs("temp://") as temp_fs, open(
        bento.path_of("bento.yaml"), "rb"
    ) as bento_yaml:
        tempdir = temp_fs.getsyspath("/")
        options = BentoInfo.from_yaml_file(bento_yaml)
        # tmpdir is our new build context.
        fs.mirror.mirror(bento._fs, temp_fs, copy_if_newer=True)

        # NOTE: dockerfile_template is already included in the
        # Dockerfile inside bento, and it is not relevant to
        # construct_containerfile. Hence it is safe to set it to None here.
        # See https://github.com/bentoml/BentoML/issues/3399.
        docker_attrs = bentoml_cattr.unstructure(options.docker)
        if (
            "dockerfile_template" in docker_attrs
            and docker_attrs["dockerfile_template"] is not None
        ):
            # NOTE: if users specify a dockerfile_template, we will
            # save it to /env/docker/Dockerfile.template. This is necessary
            # for the reconstruction of the Dockerfile.
            docker_attrs["dockerfile_template"] = "env/docker/Dockerfile.template"

        dockerfile = generate_containerfile(
            docker=DockerOptions(**docker_attrs),
            build_ctx=tempdir,
            conda=options.conda,
            bento_fs=temp_fs,
            enable_buildkit=enable_buildkit,
            add_header=add_header,
        )
        instruction.append(dockerfile)
        if features is not None:
            diff = set(features).difference(FEATURES)
            if len(diff) > 0:
                raise InvalidArgument(
                    f"Available features are: {FEATURES}. Invalid fields from provided: {diff}"
                )
            PIP_CACHE_MOUNT = (
                "--mount=type=cache,target=/root/.cache/pip " if enable_buildkit else ""
            )
            instruction.append(
                "RUN %spip install bentoml[%s]" % (PIP_CACHE_MOUNT, ",".join(features))
            )
        temp_fs.writetext(dockerfile_path, "\n".join(instruction))
        yield tempdir, temp_fs.getsyspath(dockerfile_path)


@inject
def build(
    bento_tag: Tag | str,
    backend: str,
    features: t.Sequence[str] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    **kwargs: t.Any,
):
    clean_context = contextlib.ExitStack()

    bento = _bento_store.get(bento_tag)

    builder = get_backend(backend)
    context_path, dockerfile = clean_context.enter_context(
        construct_containerfile(
            bento,
            features=features,
            enable_buildkit=enable_buildkit(builder=builder),
        )
    )
    try:
        kwargs.update({"file": dockerfile, "context_path": context_path})
        return builder.build(**kwargs)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "\nEncountered exception while trying to building image: %s",
            e,
            exc_info=sys.exc_info(),
        )
    finally:
        clean_context.close()


def register_backend(
    backend: str,
    *,
    buildkit_support: bool,
    health: t.Callable[[], bool],
    construct_build_args: t.Callable[P, Arguments],
    binary: str | None = None,
    build_cmd: t.Sequence[str] | None = None,
    env: dict[str, str] | None = None,
):
    """
    Register a custom backend, provided with a build and health check implementation.

    Args:
        backend: Name of the backend.
        buildkit_support: Whether the backend has support for BuildKit.
        health: Health check implementation. This is a callable that takes no
                argument and returns a boolean.
        construct_build_args: This is a callable that takes possible backend keyword arguments
                              and returns a list of command line arguments.
        env: Default environment variables dict for this OCI builder implementation.
        binary: Optional binary path. If not provided, the binary will use the backend name.
                Make sure that the binary is on your ``PATH``.
        build_cmd: Optional build command. If not provided, the command will be 'build'.

    Examples:

    .. code-block:: python

        from bentoml.container import register_backend

        register_backend(
            "lima",
            binary=shutil.which("limactl"),
            buildkit_support=True,
            health=buildx_health,
            construct_build_args=buildx_build_args,
            env={"DOCKER_BUILDKIT": "1"},
        )
    """
    if backend in BUILDER_REGISTRY:
        raise ValueError(f"Backend {backend} already registered.")
    if binary is None:
        binary = backend
    try:
        BUILDER_REGISTRY[backend] = OCIBuilder.create(
            binary,
            env=env,
            build_cmd=build_cmd,
            enable_buildkit=buildkit_support,
            construct_build_args=construct_build_args,
            health=health,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Failed to register backend %s: %s", backend, e, exc_info=sys.exc_info()
        )
        raise e from None


@t.overload
def health(backend: DefaultBuilder) -> bool:
    ...


@t.overload
def health(backend: str) -> bool:
    ...


def health(backend: str) -> bool:
    """
    Check if the backend is healthy.

    Args:
        backend: The name of the backend.

    .. note::

        If given backend is a type of OCIBuilder, and the backend is not registered,
        make sure to register it with ``bentoml.container.register_backend``.

    Returns:
        True if the backend is healthy, False otherwise.
    """
    return get_backend(backend).health()


@t.overload
def get_backend(backend: DefaultBuilder) -> OCIBuilder:
    ...


@t.overload
def get_backend(backend: str) -> OCIBuilder:
    ...


def get_backend(backend: str) -> OCIBuilder:
    """
    Get a given backend.

    Raises:
        ``ValueError``: If given backend is not available in backend registry.
    """
    if isinstance(backend, OCIBuilder):
        if backend not in BUILDER_REGISTRY.values():
            logger.warning(
                "Backend '%s' not registered. To use with 'bentoml.container.build', make sure to regsiter it with 'bentoml.container.register_backend'",
                backend,
            )
        return backend
    if backend not in BUILDER_REGISTRY:
        raise ValueError(
            f"Backend {backend} not registered. Available backends: {REGISTERED_BACKENDS}."
        )
    return BUILDER_REGISTRY[backend]


register_default_backends()

REGISTERED_BACKENDS = list(BUILDER_REGISTRY.keys())

__all__ = [
    "build",
    "health",
    "register_backend",
    "get_backend",
    "REGISTERED_BACKENDS",
]
