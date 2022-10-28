from __future__ import annotations

import re
import sys
import typing as t
import logging
import itertools
from typing import TYPE_CHECKING
from functools import partial

import click

if TYPE_CHECKING:
    from click import Group
    from click import Command
    from click import Context
    from click import Parameter

    P = t.ParamSpec("P")
    ClickParamType = t.Sequence[t.Any] | bool | None
    F = t.Callable[P, t.Any]

logger = logging.getLogger("bentoml")


@t.overload
def normalize_none_type(value: t.Mapping[str, t.Any]) -> t.Mapping[str, t.Any]:
    ...


@t.overload
def normalize_none_type(value: ClickParamType) -> ClickParamType:
    ...


def normalize_none_type(
    value: ClickParamType | t.Mapping[str, t.Any]
) -> ClickParamType | t.Mapping[str, t.Any]:
    if isinstance(value, (tuple, list, set, str)) and len(value) == 0:
        return
    if isinstance(value, dict):
        return {k: normalize_none_type(v) for k, v in value.items()}
    return value


# NOTE: This is the key we use to store the transformed options in the CLI context.
_MEMO_KEY = "_memoized"


class CompatibleOption(click.Option):
    def __init__(
        self,
        *param_decls: t.Any,
        obsolete: bool = False,
        deprecated: bool = False,
        **attrs: t.Any,
    ):
        if obsolete and deprecated:
            raise ValueError("'obsolete' and 'deprecated' are mutually exclusive.")
        super().__init__(*param_decls, **attrs)

        self.obsolete = obsolete
        self.deprecated = deprecated


def compatible_option(*param_decls: str, **attrs: t.Any):
    """
    Make a new ``@click.option`` decorator that provides backwards compatibility.

    This decorator extends the default ``@click.option`` decorator by adding three
    additional parameters:

        * ``obsolete``: a bool determine whether the given option is obsolete.
        * ``equivalent``: a tuple of (new_option, example_usage) that specifies
                          the new option and its example usage that replaces the given option.

        * ``deprecated``: a bool determine whether the given option is deprecated.

    .. note::

       The ``obsolete`` and ``deprecated`` options are mutually exclusive.
    """
    obsolete = attrs.pop("obsolete", False)
    deprecated = attrs.pop("deprecated", False)
    equivalent: tuple[str, str] = attrs.pop("equivalent", ())

    prepend_msg = ""
    if deprecated:
        prepend_msg = "DEPRECATED (will be removed in future release):"
    if obsolete:
        assert (
            equivalent is not None and len(equivalent) == 2
        ), "'equivalent' must be a tuple of (new_option, usage)"
        prepend_msg = "(Equivalent to --%s %s):" % (*equivalent,)

    def obsolete_callback(ctx: Context, param: Parameter, value: ClickParamType):
        # NOTE: We want to transform memoized options to tuple[str] | bool | None

        assert param.name is not None, "argument name must be set."

        opt = param.opts[0]
        # This callback handles transformation from old args to new format value.
        # This callback will also print a warning message to stderr.
        assert len(equivalent) == 2, "must be a tuple of (new_option, usage)"
        # ensure that our memoized are available under CLI context.
        if _MEMO_KEY not in ctx.params:
            # By default, multiple options are stored as a tuple.
            # our memoized options are stored as a dict.
            ctx.params[_MEMO_KEY] = {}

        # default can be a callable. We only care about the result.
        default_value = param.get_default(ctx)

        # if given param.name is not in the memoized options, we need to create them.
        if param.name not in ctx.params[_MEMO_KEY]:
            # Initialize the memoized options with default value.
            ctx.params[_MEMO_KEY][param.name] = default_value
            if param.multiple:
                # if given param is a multiple option, we need to make sure that
                # we create an empty tuple
                ctx.params[_MEMO_KEY][param.name] = ()

        # We need to run transformation here such that we don't have to deal with
        # nested value.
        value = normalize_none_type(value)
        if value is not None:
            # Only warning if given value is different from the default.
            # Since we are going to transform default value from old options to
            # the kwargs map, there is no need to bloat logs.
            if value != default_value:
                msg = "is now obsolete, use the equivalent"
                if isinstance(value, bool):
                    logger.warning(
                        "'%s' %s '--%s %s' instead.",
                        opt,
                        msg,
                        equivalent[0],
                        opt.lstrip("--"),
                    )
                elif isinstance(value, tuple):
                    obsolete_format = " ".join(map(lambda s: "%s=%s" % (opt, s), value))
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
            if isinstance(value, bool):
                ctx.params[_MEMO_KEY][param.name] = value
            elif isinstance(value, tuple):
                assert param.multiple
                ctx.params[_MEMO_KEY][param.name] += tuple(
                    map(lambda s: "%s=%s" % (opt, s), value)
                )
            else:
                assert isinstance(value, str)
                ctx.params[_MEMO_KEY][param.name] = ("%s=%s" % (opt, value),)

    def deprecated_callback(ctx: Context, param: Parameter, value: ClickParamType):
        if value is not None and value:
            logger.warning(
                "'%s' is now deprecated and will be removed in future release.",
                param.opts[0],
            )

    def decorator(f: F[t.Any]) -> t.Callable[[F[t.Any]], Command]:
        # We will set default value to None if 'default' is not set.
        msg = attrs.pop("help", None)
        assert msg is not None, "'help' is required."
        attrs.setdefault("help", " ".join([prepend_msg, msg]))
        attrs.setdefault("expose_value", True)
        attrs.setdefault("obsolete", obsolete)
        attrs.setdefault("deprecated", deprecated)
        if obsolete:
            attrs.setdefault("callback", obsolete_callback)
        if deprecated:
            attrs.setdefault("callback", deprecated_callback)
            attrs.setdefault("default", None)
        return click.option(*param_decls, cls=CompatibleOption, **attrs)(f)

    return decorator


deprecated_option = partial(compatible_option, deprecated=True)
obsolete_option = partial(compatible_option, obsolete=True)


def deprecated_options_group(f: F[t.Any]):
    deprecs = [
        deprecated_option(
            "--allow",
            multiple=True,
            help="Allow extra privileged entitlement (e.g., 'network.host', 'security.insecure').",
        ),
        deprecated_option(
            "--build-context",
            multiple=True,
            help="Additional build contexts (e.g., name=path).",
        ),
        deprecated_option(
            "--builder",
            type=click.STRING,
            help="Override the configured builder instance.",
        ),
        deprecated_option(
            "--cache-to",
            multiple=True,
            help="Cache export destinations (e.g., 'user/app:cache', 'type=local,dest=path/to/dir').",
        ),
        deprecated_option(
            "--load",
            is_flag=True,
            default=False,
            help="Shorthand for '--output=type=docker'. Note that '--push' and '--load' are mutually exclusive.",
        ),
        deprecated_option(
            "--push",
            is_flag=True,
            default=False,
            help="Shorthand for '--output=type=registry'. Note that '--push' and '--load' are mutually exclusive.",
        ),
        deprecated_option(
            "--no-cache-filter",
            multiple=True,
            help="Do not cache specified stages.",
        ),
        deprecated_option(
            "--metadata-file",
            type=click.STRING,
            help="Write build result metadata to the file.",
        ),
        deprecated_option("--shm-size", help="Size of '/dev/shm'."),
    ]
    for options in reversed(deprecs):
        f = options(f)
    return f


def obsolete_options_group(f: F[t.Any]):
    obsoletes = [
        obsolete_option(
            "--add-host",
            equivalent=("opt", "add-host=host:ip"),
            multiple=True,
            metavar="[HOST:IP,]",
            help="Add a custom host-to-IP mapping.",
        ),
        obsolete_option(
            "--build-arg",
            equivalent=("opt", "build-arg=FOO=bar"),
            multiple=True,
            metavar="[KEY=VALUE,]",
            help="Set build-time variables.",
        ),
        obsolete_option(
            "--cache-from",
            equivalent=("opt", "cache-from=user/app:cache"),
            multiple=True,
            default=None,
            metavar="[NAME|type=TYPE[,KEY=VALUE]]",
            help="External cache sources (e.g.: 'type=local,src=path/to/dir').",
        ),
        obsolete_option(
            "--iidfile",
            type=click.STRING,
            equivalent=("opt", "iidfile=/path/to/file"),
            default=None,
            metavar="PATH",
            help="Write the image ID to the file.",
        ),
        obsolete_option(
            "--label",
            equivalent=("opt", "label=io.container.capabilities=baz"),
            multiple=True,
            metavar="[NAME|[KEY=VALUE],]",
            help="Set metadata for an image. This is equivalent to LABEL in given Dockerfile.",
        ),
        obsolete_option(
            "--network",
            type=click.STRING,
            equivalent=("opt", "network=default"),
            metavar="default|VALUE",
            default=None,
            help="Set the networking mode for the 'RUN' instructions during build (default 'default').",
        ),
        obsolete_option(
            "--no-cache",
            equivalent=("opt", "no-cache"),
            is_flag=True,
            default=False,
            help="Do not use cache when building the image.",
        ),
        obsolete_option(
            "--output",
            equivalent=("opt", "output=type=local,dest=/path/to/dir"),
            multiple=True,
            default=None,
            metavar="[PATH,-,type=TYPE[,KEY=VALUE]",
            help="Output destination.",
        ),
        obsolete_option(
            "--platform",
            equivalent=("opt", "platform=linux/amd64,linux/arm/v7"),
            default=None,
            metavar="VALUE[,VALUE]",
            multiple=True,
            type=click.STRING,
            help="Set target platform for build.",
        ),
        obsolete_option(
            "--progress",
            equivalent=("opt", "progress=auto"),
            default="auto",
            type=click.Choice(["auto", "tty", "plain"]),
            metavar="VALUE",
            help="Set type of progress output. Use plain to show container output if BENTOML_DEBUG is set.",
        ),
        obsolete_option(
            "--pull",
            equivalent=("opt", "pull"),
            is_flag=True,
            default=False,
            help="Always attempt to pull all referenced images.",
        ),
        obsolete_option(
            "--secret",
            multiple=True,
            equivalent=("opt", "secret=id=aws,src=$HOME/.aws/crendentials"),
            metavar="[type=TYPE[,KEY=VALUE]",
            default=None,
            help="Secret to expose to the build.",
        ),
        obsolete_option(
            "--ssh",
            type=click.STRING,
            equivalent=("opt", "ssh=default=$SSH_AUTH_SOCK"),
            metavar="default|<id>[=<socket>|<key>[,<key>]]",
            default=None,
            help="SSH agent socket or keys to expose to the build. See https://docs.docker.com/engine/reference/commandline/buildx_build/#ssh",
        ),
        obsolete_option(
            "--target",
            equivalent=("opt", "target=release-stage"),
            metavar="VALUE",
            type=click.STRING,
            default=None,
            help="Set the target build stage to build.",
        ),
        obsolete_option(
            "--cgroup-parent",
            equivalent=("opt", "cgroup-parent=mygroup"),
            type=click.STRING,
            default=None,
            help="Optional parent cgroup for the container.",
        ),
        obsolete_option(
            "--ulimit",
            equivalent=("opt", "ulimit=nofile=1024:1024"),
            type=click.STRING,
            metavar="<type>=<soft limit>[:<hard limit>]",
            default=None,
            help="Ulimit options.",
        ),
    ]
    for options in reversed(obsoletes):
        f = options(f)
    return f


def opt_callback(ctx: Context, param: Parameter, value: ClickParamType):
    # NOTE: our new options implementation will have the following format:
    #   --opt ARG[=|:]VALUE[,VALUE] (e.g., --opt key1=value1,value2 --opt key2:value3/value4:hello)
    # Argument and values per --opt has one-to-one or one-to-many relationship,
    # separated by '=' or ':'.
    # TODO: We might also want to support the following format:
    #  --opt arg1 value1 arg2 value2 --opt arg3 value3
    #  --opt arg1=value1,arg2=value2 --opt arg3:value3

    assert param.multiple, "Only use this callback when multiple=True."
    if _MEMO_KEY not in ctx.params:
        # By default, multiple options are stored as a tuple.
        # our memoized options are stored as a dict.
        ctx.params[_MEMO_KEY] = {}

    if param.name not in ctx.params[_MEMO_KEY]:
        ctx.params[_MEMO_KEY][param.name] = ()

    value = normalize_none_type(value)
    if value is not None:
        normalized: t.Callable[[str], str] = lambda opt: opt.replace("-", "_")
        if isinstance(value, tuple):
            for opt in value:
                o, *val = re.split(r"=|:", opt, maxsplit=1)
                norm = normalized(o)
                if len(val) == 0:
                    # --opt bool
                    ctx.params[_MEMO_KEY][norm] = True
                else:
                    # --opt key=value
                    ctx.params[_MEMO_KEY].setdefault(norm, ())
                    ctx.params[_MEMO_KEY][norm] += ("--%s=%s" % (o, *val),)
        else:
            # Note that if we weren't able to parse the value, then we memoized
            # to _MEMO_KEY[opt]
            logger.warning("Failed to parse '%s' with value %s", param.name, value)
            ctx.params[_MEMO_KEY][param.name] += (value,)


def add_containerize_command(cli: Group) -> None:
    from bentoml.bentos import containerize as containerize_bento
    from bentoml.bentos import determine_container_tag
    from bentoml._internal import container
    from bentoml_cli.utils import kwargs_transformers
    from bentoml_cli.utils import validate_container_tag
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command()
    @click.argument("bento_tag", type=click.STRING, metavar="BENTO:TAG")
    @click.option(
        "-t",
        "--image-tag",
        "--docker-image-tag",
        help="Name and optionally a tag (format: 'name:tag') for building container, defaults to the bento tag.",
        required=False,
        callback=validate_container_tag,
        multiple=True,
    )
    @click.option(
        "--backend",
        help=f"Define OCI backend. Available: {', '.join(container.registered_backends)}",
        required=False,
        default="docker",
        type=click.Choice(container.registered_backends),
    )
    @click.option(
        "--opt",
        help="Define options for given backend. (format: '--opt target=foo --opt build_arg=foo=BAR,build_arg=baz=RAR --opt no-cache')",
        required=False,
        multiple=True,
        callback=opt_callback,
        metavar="ARG=VALUE[,ARG=VALUE]",
    )
    @click.option(
        "--enable-features",
        multiple=True,
        nargs=1,
        metavar="FEATURE[,FEATURE]",
        help=f"Enable additional BentoML features. Available features are: {', '.join(container.FEATURES)}.",
    )
    @obsolete_options_group
    @deprecated_options_group
    @kwargs_transformers(transformer=normalize_none_type)
    def containerize(  # type: ignore
        bento_tag: str,
        image_tag: tuple[str] | None,
        backend: str,
        enable_features: tuple[str] | None,
        _memoized: dict[str, ClickParamType],
        **kwargs: t.Any,
    ) -> None:
        """Containerizes given Bento into an OCI-compliant container, with any of your favorite OCI builder.

        \b
        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        \b
        ``bentoml containerize`` command also supports the use of the ``latest`` tag
        which will automatically use the last built version of your Bento.

        \b
        You can provide a tag for the image built by Bento using the
        ``--image-tag`` flag.

        \b
        You can also prefixing the tag with a hostname for the repository you wish
        to push to.

        .. code-block:: bash

           $ bentoml containerize iris_classifier:latest -t repo-address.com:username/iris

        This would build an image called ``username/iris:latest`` and push it to docker repository at repo-address.com.

        \b
        By default, the ``containerize`` command will use the current credentials provided by each backend implementation.

        \b
        ``bentoml containerize`` also leverage BuildKit features such as multi-node
        builds for cross-platform images, more efficient caching, and more secure builds.
        BuildKit will be enabled by default by each backend implementation.
        """

        # We should already handle all options in callback to memoized, but just in case...
        filtered = dict(filter(lambda item: item[1] is not None, kwargs.items()))
        if len(filtered) != 0:
            logger.debug("Unexpected arguments: %s and failed to parse them.", filtered)
        logger.debug("Memoized: %s", _memoized)

        features = None
        if enable_features:
            features = tuple(
                itertools.chain.from_iterable(
                    map(lambda s: s.split(","), enable_features)
                )
            )
        exit_code = not containerize_bento(
            bento_tag,
            backend=backend,
            image_tag=image_tag,
            features=features,
            **_memoized,
        )

        if not exit_code:
            tags = determine_container_tag(bento_tag, image_tag=image_tag)

            logger.info(
                'Successfully built docker image for "%s" with tags "%s"',
                bento_tag,
                ",".join(tags),
            )

            container_tags = ",".join(tags)
            fmt = [container_tags]
            if len(tags) > 1:
                instruction = 'To run your newly built Bento container, use one of the above tags, and pass it to "docker run". For example: "docker run -it --rm -p 3000:3000 %s serve --production".'
            else:
                instruction = 'To run your newly built Bento container, pass "%s" to "docker run". For example: "docker run -it --rm -p 3000:3000 %s serve --production".'
                fmt.append(container_tags)
            if features is not None and "grpc" in features:
                grpc_metrics_port = BentoMLContainer.grpc.metrics.port.get()
                instruction += '\nAdditionally, to run your Bento container as a gRPC server, do: "docker run -it --rm -p 3000:3000 -p %s:%s %s serve-grpc --production"'
                fmt.extend([grpc_metrics_port, grpc_metrics_port, container_tags])
            logger.info(instruction, *fmt)
        sys.exit(exit_code)
