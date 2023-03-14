from __future__ import annotations

import sys
import shutil
import typing as t
import logging
import tempfile
import itertools
import subprocess
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from click import Group
    from click import Command
    from click import Context
    from click import Parameter

    from bentoml._internal.container import DefaultBuilder

    from .utils import ClickParamType

    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]

logger = logging.getLogger("bentoml")


def compatible_option(*param_decls: str, **attrs: t.Any):
    """
    Make a new ``@click.option`` decorator that provides backwards compatibility.

    This decorator extends the default ``@click.option`` decorator by adding three
    additional parameters:

        * ``equivalent``: a tuple of (new_option, example_usage) that specifies
                          the new option and its example usage that replaces the given option.
        * ``append_msg``: a string to append a help message to the original help message.
    """
    import click

    from .utils import MEMO_KEY
    from .utils import normalize_none_type

    append_msg = attrs.pop("append_msg", "")
    equivalent: tuple[str, str] = attrs.pop("equivalent", ())
    factory = attrs.pop("factory", click)

    assert (
        equivalent is not None and len(equivalent) == 2
    ), "'equivalent' must be a tuple of (new_option, usage)"

    prepend_msg = "(Equivalent to ``--%s %s``):" % (*equivalent,)

    def obsolete_callback(ctx: Context, param: Parameter, value: ClickParamType):
        # NOTE: We want to transform memoized options to tuple[str] | bool | None

        assert param.name is not None, "argument name must be set."

        opt = param.opts[0]
        # ensure that our memoized are available under CLI context.
        if MEMO_KEY not in ctx.params:
            # By default, multiple options are stored as a tuple.
            # our memoized options are stored as a dict.
            ctx.params[MEMO_KEY] = {}

        # default can be a callable. We only care about the result.
        default_value = param.get_default(ctx)

        # We need to run transformation here such that we don't have to deal with
        # nested value.
        # NOTE: if users are using both old and new options, the new option will take
        # precedence, and the old option will be ignored.
        value = normalize_none_type(value)

        # if given param.name is not in the memoized options, we need to create them.
        if param.name not in ctx.params[MEMO_KEY]:
            # Initialize the memoized options with default value.
            ctx.params[MEMO_KEY][param.name] = () if param.multiple else default_value
            if value is not None:
                # Only warning if given value is different from the default.
                # Since we are going to transform default value from old options to
                # the kwargs map, there is no need to bloat logs.
                if value != default_value:
                    msg = "is now deprecated, use the equivalent"
                    if isinstance(value, bool):
                        logger.warning(
                            "'%s' %s '--%s %s' instead.",
                            opt,
                            msg,
                            equivalent[0],
                            opt.lstrip("--"),
                        )
                    elif isinstance(value, tuple):
                        obsolete_format = " ".join(
                            map(lambda s: "%s=%s" % (opt, s), value)
                        )
                        new_format = " ".join(
                            map(
                                lambda s: "--%s %s=%s"
                                % (equivalent[0], opt.lstrip("--"), s),
                                value,
                            )
                        )
                        logger.warning(
                            "'%s' %s '%s' instead.", obsolete_format, msg, new_format
                        )
                    else:
                        assert isinstance(value, str)
                        logger.warning(
                            "'%s=%s' %s '--%s %s=%s' instead.",
                            opt,
                            value,
                            msg,
                            equivalent[0],
                            opt.lstrip("--"),
                            value,
                        )
                if isinstance(value, (bool, str)):
                    ctx.params[MEMO_KEY][param.name] = value
                elif isinstance(value, tuple):
                    assert param.multiple
                    ctx.params[MEMO_KEY][param.name] += value
                else:
                    raise ValueError(f"Unexpected value type: {type(value)}")

    def decorator(f: F[t.Any]) -> t.Callable[[F[t.Any]], Command]:
        # We will set default value to None if 'default' is not set.
        msg = attrs.pop("help", None)
        assert msg is not None, "'help' is required."
        attrs.setdefault("help", " ".join([prepend_msg, msg, append_msg]))
        attrs.setdefault("expose_value", True)
        attrs.setdefault("callback", obsolete_callback)
        return factory.option(*param_decls, **attrs)(f)

    return decorator


def compatible_options_group(f: F[t.Any]):
    import click
    from click_option_group import optgroup

    optgroup_option = partial(compatible_option, factory=optgroup)

    compat = [
        optgroup_option(
            "--add-host",
            equivalent=("opt", "add-host=host:ip"),
            multiple=True,
            metavar="[HOST:IP,]",
            help="Add a custom host-to-IP mapping.",
        ),
        optgroup_option(
            "--build-arg",
            equivalent=("opt", "build-arg=FOO=bar"),
            multiple=True,
            metavar="[KEY=VALUE,]",
            help="Set build-time variables.",
        ),
        optgroup_option(
            "--cache-from",
            equivalent=("opt", "cache-from=user/app:cache"),
            multiple=True,
            default=None,
            metavar="[NAME|type=TYPE[,KEY=VALUE]]",
            help="External cache sources (e.g.: ``type=local,src=path/to/dir``).",
        ),
        optgroup_option(
            "--iidfile",
            type=click.STRING,
            equivalent=("opt", "iidfile=/path/to/file"),
            default=None,
            metavar="PATH",
            help="Write the image ID to the file.",
        ),
        optgroup_option(
            "--label",
            equivalent=("opt", "label=io.container.capabilities=baz"),
            multiple=True,
            metavar="[NAME|[KEY=VALUE],]",
            help="Set metadata for an image. This is equivalent to LABEL in given Dockerfile.",
        ),
        optgroup_option(
            "--network",
            type=click.STRING,
            equivalent=("opt", "network=default"),
            metavar="default|VALUE",
            default=None,
            help="Set the networking mode for the ``RUN`` instructions during build (default ``default``).",
        ),
        optgroup_option(
            "--no-cache",
            equivalent=("opt", "no-cache"),
            is_flag=True,
            default=False,
            help="Do not use cache when building the image.",
        ),
        optgroup_option(
            "--output",
            equivalent=("opt", "output=type=local,dest=/path/to/dir"),
            multiple=True,
            default=None,
            metavar="[PATH,-,type=TYPE[,KEY=VALUE]",
            help="Output destination.",
        ),
        optgroup_option(
            "--platform",
            equivalent=("opt", "platform=linux/amd64,linux/arm/v7"),
            default=None,
            metavar="VALUE[,VALUE]",
            multiple=True,
            type=click.STRING,
            help="Set target platform for build.",
        ),
        optgroup_option(
            "--progress",
            equivalent=("opt", "progress=auto"),
            default=None,
            type=click.Choice(["auto", "tty", "plain"]),
            metavar="VALUE",
            help="Set type of progress output. Use plain to show container output.",
        ),
        optgroup_option(
            "--pull",
            equivalent=("opt", "pull"),
            is_flag=True,
            default=False,
            help="Always attempt to pull all referenced images.",
        ),
        optgroup_option(
            "--secret",
            multiple=True,
            equivalent=("opt", "secret=id=aws,src=$HOME/.aws/crendentials"),
            metavar="[type=TYPE[,KEY=VALUE]",
            default=None,
            help="Secret to expose to the build.",
        ),
        optgroup_option(
            "--ssh",
            type=click.STRING,
            equivalent=("opt", "ssh=default=$SSH_AUTH_SOCK"),
            metavar="default|<id>[=<socket>|<key>[,<key>]]",
            default=None,
            help="SSH agent socket or keys to expose to the build. See https://docs.docker.com/engine/reference/commandline/buildx_build/#ssh",
        ),
        optgroup_option(
            "--target",
            equivalent=("opt", "target=release-stage"),
            metavar="VALUE",
            type=click.STRING,
            default=None,
            help="Set the target build stage to build.",
        ),
    ]
    for options in reversed(compat):
        f = options(f)
    return f


def buildx_options_group(f: F[t.Any]):
    import click
    from click_option_group import optgroup

    optgroup_option = partial(compatible_option, factory=optgroup)

    buildx = [
        optgroup_option(
            "--allow",
            equivalent=("opt", "allow=network.host"),
            multiple=True,
            metavar="ENTITLEMENT",
            help="Allow extra privileged entitlement (e.g., ``network.host``, ``security.insecure``).",
        ),
        optgroup_option(
            "--build-context",
            equivalent=("opt", "build-context=project=path/to/project/source"),
            multiple=True,
            metavar="name=VALUE",
            help="Additional build contexts (e.g., ``name=path``).",
        ),
        optgroup_option(
            "--builder",
            equivalent=("opt", "builder=default"),
            type=click.STRING,
            help="Override the configured builder instance. Same as ``docker buildx --builder``",
        ),
        optgroup_option(
            "--cache-to",
            multiple=True,
            equivalent=("opt", "cache-to=type=registry,ref=user/app"),
            metavar="[NAME|type=TYPE[,KEY=VALUE]]",
            help="Cache export destinations (e.g., ``user/app:cache``, ``type=local,dest=path/to/dir``).",
        ),
        optgroup_option(
            "--load",
            is_flag=True,
            equivalent=("opt", "load"),
            default=False,
            help="Shorthand for ``--output=type=docker``. Note that ``--push`` and ``--load`` are mutually exclusive.",
        ),
        optgroup_option(
            "--push",
            is_flag=True,
            equivalent=("opt", "push"),
            default=False,
            help="Shorthand for ``--output=type=registry``. Note that ``--push`` and ``--load`` are mutually exclusive.",
        ),
        optgroup_option(
            "--quiet",
            is_flag=True,
            equivalent=("opt", "quiet"),
            default=False,
            help="Suppress the build output and print image ID on success",
        ),
        optgroup_option(
            "--cgroup-parent",
            equivalent=("opt", "cgroup-parent=cgroupv2"),
            type=click.STRING,
            default=None,
            help="Optional parent cgroup for the container.",
        ),
        optgroup_option(
            "--ulimit",
            equivalent=("opt", "ulimit=nofile=1024:1024"),
            type=click.STRING,
            metavar="<type>=<soft limit>[:<hard limit>]",
            default=None,
            help="Ulimit options.",
        ),
        optgroup_option(
            "--no-cache-filter",
            equivalent=("opt", "no-cache-filter"),
            multiple=True,
            help="Do not cache specified stages.",
        ),
        optgroup_option(
            "--metadata-file",
            type=click.STRING,
            equivalent=("opt", "metadata-file=/path/to/file"),
            help="Write build result metadata to the file.",
        ),
        optgroup_option(
            "--shm-size",
            equivalent=("opt", "shm-size=8192Mb"),
            help="Size of ``/dev/shm``.",
        ),
    ]
    for options in reversed(buildx):
        f = options(f)
    return f


def add_containerize_command(cli: Group) -> None:
    import click
    from click_option_group import optgroup

    from bentoml import container
    from bentoml_cli.utils import opt_callback
    from bentoml_cli.utils import kwargs_transformers
    from bentoml_cli.utils import normalize_none_type
    from bentoml_cli.utils import validate_container_tag
    from bentoml.exceptions import BentoMLException
    from bentoml._internal.container import FEATURES
    from bentoml._internal.container import enable_buildkit
    from bentoml._internal.container import REGISTERED_BACKENDS
    from bentoml._internal.container import determine_container_tag
    from bentoml._internal.configuration import get_debug_mode

    @cli.command()
    @click.argument("bento_tag", type=click.STRING, metavar="BENTO:TAG")
    @click.option(
        "-t",
        "--image-tag",
        "--docker-image-tag",
        metavar="NAME:TAG",
        help="Name and optionally a tag (format: ``name:tag``) for building container, defaults to the bento tag.",
        required=False,
        callback=validate_container_tag,
        multiple=True,
    )
    @click.option(
        "--backend",
        help="Define builder backend. Available: {}".format(
            ", ".join(map(lambda s: f"``{s}``", REGISTERED_BACKENDS))
        ),
        required=False,
        default="docker",
        envvar="BENTOML_CONTAINERIZE_BACKEND",
        type=click.STRING,
    )
    @click.option(
        "--enable-features",
        multiple=True,
        nargs=1,
        metavar="FEATURE[,FEATURE]",
        help="Enable additional BentoML features. Available features are: {}.".format(
            ", ".join(map(lambda s: f"``{s}``", FEATURES))
        ),
    )
    @click.option(
        "--opt",
        help="Define options for given backend. (format: ``--opt target=foo --opt build-arg=foo=BAR --opt iidfile:/path/to/file --opt no-cache``)",
        required=False,
        multiple=True,
        callback=opt_callback,
        metavar="ARG=VALUE[,ARG=VALUE]",
    )
    @click.option(
        "--run-as-root",
        help="Whether to run backend with as root.",
        is_flag=True,
        required=False,
        default=False,
        type=bool,
    )
    @optgroup.group(
        "Equivalent options", help="Equivalent option group for backward compatibility"
    )
    @compatible_options_group
    @optgroup.group("Buildx options", help="[Only available with '--backend=buildx']")
    @buildx_options_group
    @kwargs_transformers(transformer=normalize_none_type)
    def containerize(  # type: ignore
        bento_tag: str,
        image_tag: tuple[str] | None,
        backend: DefaultBuilder,
        enable_features: tuple[str] | None,
        run_as_root: bool,
        _memoized: dict[str, t.Any],
        **kwargs: t.Any,  # pylint: disable=unused-argument
    ) -> None:
        """Containerizes given Bento into an OCI-compliant container, with any given OCI builder.

        \b

        ``BENTO`` is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: ``iris_classifier:v1.2.0``

        \b
        ``bentoml containerize`` command also supports the use of the ``latest`` tag
        which will automatically use the last built version of your Bento.

        \b
        You can provide a tag for the image built by Bento using the
        ``--image-tag`` flag.

        \b
        You can also prefixing the tag with a hostname for the repository you wish to push to.

        .. code-block:: bash

           $ bentoml containerize iris_classifier:latest -t repo-address.com:username/iris

        This would build an image called ``username/iris:latest`` and push it to docker repository at ``repo-address.com``.

        \b
        By default, the ``containerize`` command will use the current credentials provided by each backend implementation.

        \b
        ``bentoml containerize`` also leverage BuildKit features such as multi-node
        builds for cross-platform images, more efficient caching, and more secure builds.
        BuildKit will be enabled by default by each backend implementation. To opt-out of BuildKit, set ``DOCKER_BUILDKIT=0``:

        .. code-block:: bash

            $ DOCKER_BUILDKIT=0 bentoml containerize iris_classifier:latest

        \b
        ``bentoml containerize`` is backend-agnostic, meaning that it can use any OCI-compliant builder to build the container image.
        By default, it will use the ``docker`` backend, which will use the local Docker daemon to build the image.
        You can use the ``--backend`` flag to specify a different backend. Note that backend flags will now be unified via ``--opt``.

        ``--opt`` accepts multiple flags, and it works with all backend options.

        \b
        To pass a boolean flag (e.g: ``--no-cache``), use ``--opt no-cache``.

        \b
        To pass a key-value pair (e.g: ``--build-arg FOO=BAR``), use ``--opt build-arg=FOO=BAR`` or ``--opt build-arg:FOO=BAR``. Make sure to quote the value if it contains spaces.

        \b
        To pass a argument with one value (path, integer, etc) (e.g: ``--cgroup-parent cgroupv2``), use ``--opt cgroup-parent=cgroupv2`` or ``--opt cgroup-parent:cgroupv2``.
        """

        # we can filter out all options that are None or False.
        _memoized = {k: v for k, v in _memoized.items() if v}
        logger.debug("Memoized: %s", _memoized)

        # Run healthcheck before containerizing
        # build will also run healthcheck, but we want to fail early.
        if not container.health(backend):
            raise BentoMLException("Failed to use backend %s." % backend)

        # --progress is not available without BuildKit.
        if not enable_buildkit(backend=backend) and "progress" in _memoized:
            _memoized.pop("progress")
        elif get_debug_mode():
            _memoized["progress"] = "plain"
        _memoized["_run_as_root"] = run_as_root

        features: tuple[str] | None = None
        if enable_features:
            features = tuple(
                itertools.chain.from_iterable(
                    map(lambda s: s.split(","), enable_features)
                )
            )
        result = container.build(
            bento_tag,
            backend=backend,
            image_tag=image_tag,
            features=features,
            **_memoized,
        )

        if result is not None:
            tags = determine_container_tag(bento_tag, image_tag=image_tag)

            container_runtime = backend
            # This logic determines some of the output messages.
            if backend == "buildah":
                if shutil.which("podman") is None:
                    logger.warning(
                        "'buildah' are only used to build the image. To run the image, 'podman' is required to be installed (podman not found under PATH). See https://podman.io/getting-started/installation."
                    )
                    sys.exit(0)
                else:
                    container_runtime = "podman"
            elif backend == "buildx":
                # buildx is a syntatic sugar for docker buildx build
                container_runtime = "docker"
            elif backend == "buildctl":
                if shutil.which("docker") is not None:
                    container_runtime = "docker"
                elif shutil.which("podman") is not None:
                    container_runtime = "podman"
                elif shutil.which("nerdctl") is not None:
                    container_runtime = "nerdctl"
                else:
                    click.echo(
                        click.style(
                            "To load image built with 'buildctl' requires one of "
                            "docker, podman, nerdctl (None are found in PATH). "
                            "Make sure they are visible to PATH and try again.",
                            fg="yellow",
                        ),
                        color=True,
                    )
                    sys.exit(0)
                if "output" not in _memoized:
                    tmp_path = tempfile.gettempdir()
                    type_prefix = "type=oci,name="
                    if container_runtime == "docker":
                        type_prefix = "type=docker,name=docker.io/"
                    click.echo(
                        click.style(
                            "Autoconfig is now deprecated and will be removed "
                            "in the next future release. We recommend setting "
                            "output to a tarfile should you wish to load the "
                            "image locally. For example:",
                            fg="yellow",
                        ),
                        color=True,
                    )
                    click.echo(
                        click.style(
                            f"    bentoml containerize {bento_tag} --backend buildctl --opt output={type_prefix}{tags[0]},dest={tmp_path}/{tags[0].replace(':', '_')}.tar\n"
                            f"    {container_runtime} load -i {tmp_path}/{tags[0].replace(':', '_')}.tar",
                            fg="yellow",
                        ),
                        color=True,
                    )
                    o = subprocess.check_output(
                        [container_runtime, "load"], input=result
                    )
                    if get_debug_mode():
                        click.echo(o.decode("utf-8").strip())
                    sys.exit(0)
                return result

            click.echo(
                f'Successfully built Bento container for "{bento_tag}" with tag(s) "{",".join(tags)}"',
            )
            instructions = (
                "To run your newly built Bento container, run:\n"
                + f"    {container_runtime} run -it --rm -p 3000:3000 {tags[0]} serve --production\n"
            )

            if features is not None and any(
                has_grpc in features
                for has_grpc in ("all", "grpc", "grpc-reflection", "grpc-channelz")
            ):
                instructions += (
                    "To serve with gRPC instead, run:\n"
                    + f"    {container_runtime} run -it -p 3000:3000 -p 3001:3001 {tags[0]} serve-grpc --production\n"
                )
            click.echo(instructions, nl=False)
            raise SystemExit(0)
        raise SystemExit(1)
