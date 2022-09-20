from __future__ import annotations

import sys
import typing as t
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from click import Group

    from bentoml._internal.bento import BentoStore

    F = t.Callable[..., t.Any]


def containerize_transformer(
    value: t.Iterable[str] | str | bool | None,
) -> t.Iterable[str] | str | bool | None:
    if value is None:
        return
    if isinstance(value, tuple) and not value:
        return
    return value


def add_containerize_command(cli: Group) -> None:
    import click
    from simple_di import inject
    from simple_di import Provide

    from bentoml.bentos import FEATURES
    from bentoml.bentos import containerize as containerize_bento
    from bentoml_cli.utils import kwargs_transformers
    from bentoml_cli.utils import validate_docker_tag
    from bentoml._internal.configuration import get_debug_mode
    from bentoml._internal.configuration.containers import BentoMLContainer

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-t",
        "--docker-image-tag",
        help="Name and optionally a tag (format: 'name:tag'), defaults to bento tag.",
        required=False,
        callback=validate_docker_tag,
        multiple=True,
    )
    @click.option(
        "--add-host",
        multiple=True,
        help="Add a custom host-to-IP mapping (format: 'host:ip').",
    )
    @click.option(
        "--allow",
        multiple=True,
        default=None,
        help="Allow extra privileged entitlement (e.g., 'network.host', 'security.insecure').",
    )
    @click.option("--build-arg", multiple=True, help="Set build-time variables.")
    @click.option(
        "--build-context",
        multiple=True,
        help="Additional build contexts (e.g., name=path).",
    )
    @click.option(
        "--builder",
        type=click.STRING,
        default=None,
        help="Override the configured builder instance.",
    )
    @click.option(
        "--cache-from",
        multiple=True,
        default=None,
        help="External cache sources (e.g., 'user/app:cache', 'type=local,src=path/to/dir').",
    )
    @click.option(
        "--cache-to",
        multiple=True,
        default=None,
        help="Cache export destinations (e.g., 'user/app:cache', 'type=local,dest=path/to/dir').",
    )
    @click.option(
        "--cgroup-parent",
        type=click.STRING,
        default=None,
        help="Optional parent cgroup for the container.",
    )
    @click.option(
        "--iidfile",
        type=click.STRING,
        default=None,
        help="Write the image ID to the file.",
    )
    @click.option("--label", multiple=True, help="Set metadata for an image.")
    @click.option(
        "--load",
        is_flag=True,
        default=False,
        help="Shorthand for '--output=type=docker'. Note that '--push' and '--load' are mutually exclusive.",
    )
    @click.option(
        "--metadata-file",
        type=click.STRING,
        default=None,
        help="Write build result metadata to the file.",
    )
    @click.option(
        "--network",
        type=click.STRING,
        default=None,
        help="Set the networking mode for the 'RUN' instructions during build (default 'default').",
    )
    @click.option(
        "--no-cache",
        is_flag=True,
        default=False,
        help="Do not use cache when building the image.",
    )
    @click.option(
        "--no-cache-filter",
        multiple=True,
        help="Do not cache specified stages.",
    )
    @click.option(
        "--output",
        multiple=True,
        default=None,
        help="Output destination (format: 'type=local,dest=path').",
    )
    @click.option(
        "--platform", default=None, multiple=True, help="Set target platform for build."
    )
    @click.option(
        "--progress",
        default="auto",
        type=click.Choice(["auto", "tty", "plain"]),
        help="Set type of progress output ('auto', 'plain', 'tty'). Use plain to show container output.",
    )
    @click.option(
        "--pull",
        is_flag=True,
        default=False,
        help="Always attempt to pull all referenced images.",
    )
    @click.option(
        "--push",
        is_flag=True,
        default=False,
        help="Shorthand for '--output=type=registry'. Note that '--push' and '--load' are mutually exclusive.",
    )
    @click.option(
        "--secret",
        multiple=True,
        default=None,
        help="Secret to expose to the build (format: 'id=mysecret[,src=/local/secret]').",
    )
    @click.option("--shm-size", default=None, help="Size of '/dev/shm'.")
    @click.option(
        "--ssh",
        type=click.STRING,
        default=None,
        help="SSH agent socket or keys to expose to the build (format: 'default|<id>[=<socket>|<key>[,<key>]]').",
    )
    @click.option(
        "--target",
        type=click.STRING,
        default=None,
        help="Set the target build stage to build.",
    )
    @click.option(
        "--ulimit", type=click.STRING, default=None, help="Ulimit options (default [])."
    )
    @click.option(
        "--enable-features",
        multiple=True,
        nargs=1,
        metavar="[features,]",
        help=f"Enable additional BentoML features. Available features are: {', '.join(FEATURES)}.",
    )
    @kwargs_transformers(transformer=containerize_transformer)
    @inject
    def containerize(  # type: ignore
        bento_tag: str,
        docker_image_tag: tuple[str],
        add_host: tuple[str],
        allow: tuple[str],
        build_arg: tuple[str],
        build_context: tuple[str],
        builder: str,
        cache_from: tuple[str],
        cache_to: tuple[str],
        cgroup_parent: str,
        iidfile: str,
        label: tuple[str],
        load: bool,
        network: str,
        metadata_file: str,
        no_cache: bool,
        no_cache_filter: tuple[str],
        output: tuple[str],
        platform: tuple[str],
        progress: t.Literal["auto", "tty", "plain"],
        pull: bool,
        push: bool,
        secret: tuple[str],
        shm_size: str,
        ssh: str,
        target: str,
        ulimit: str,
        enable_features: tuple[str],
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    ) -> None:
        """Containerizes given Bento into a ready-to-use Docker image.

        \b
        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        'bentoml containerize' command also supports the use of the 'latest' tag
        which will automatically use the last built version of your Bento.

        You can provide a tag for the image built by Bento using the
        '--docker-image-tag' flag. Additionally, you can provide a '--push' flag,
        which will push the built image to the Docker repository specified by the
        image tag.

        You can also prefixing the tag with a hostname for the repository you wish
        to push to.
        e.g. 'bentoml containerize IrisClassifier:latest --push --tag
        repo-address.com:username/iris' would build a Docker image called
        'username/iris:latest' and push that to docker repository at repo-address.com.

        By default, the 'containerize' command will use the current credentials
        provided by Docker daemon.

        'bentoml containerize' also uses Docker Buildx as backend, in place for normal 'docker build'.
        By doing so, BentoML will leverage Docker Buildx features such as multi-node
        builds for cross-platform images, Full BuildKit capabilities with all of the
        familiar UI from 'docker build'.
        """
        from bentoml._internal.utils import buildx

        logger = logging.getLogger("bentoml")

        # run health check whether buildx is install locally
        buildx.health()

        add_hosts = {}
        if add_host:
            for host in add_host:
                host_name, ip = host.split(":")
                add_hosts[host_name] = ip

        allow_ = []
        if allow:
            allow_ = list(allow)

        build_args = {}
        if build_arg:
            for build_arg_str in build_arg:
                key, value = build_arg_str.split("=")
                build_args[key] = value

        build_context_ = {}
        if build_context:
            for build_context_str in build_context:
                key, value = build_context_str.split("=")
                build_context_[key] = value

        labels = {}
        if label:
            for label_str in label:
                key, value = label_str.split("=")
                labels[key] = value

        output_: dict[str, t.Any] | None = None
        if output:
            output_ = {}
            for arg in output:
                if "," in arg:
                    for val in arg.split(","):
                        k, v = val.split("=")
                        output_[k] = v
                key, value = arg.split("=")
                output_[key] = value

        load = True
        if platform and len(platform) > 1:
            if not push:
                logger.warning(
                    "Multiple '--platform' arguments were found. Make sure to also use '--push' to push images to a repository or generated images will not be saved. For more information, see https://docs.docker.com/engine/reference/commandline/buildx_build/#load."
                )
        if push:
            load = False
        if get_debug_mode():
            progress = "plain"

        exit_code = not containerize_bento(
            bento_tag,
            docker_image_tag=docker_image_tag,
            # containerize options
            features=enable_features,
            # docker options
            add_host=add_hosts,
            allow=allow_,
            build_args=build_args,
            build_context=build_context_,
            builder=builder,
            cache_from=cache_from,
            cache_to=cache_to,
            cgroup_parent=cgroup_parent,
            iidfile=iidfile,
            labels=labels,
            load=load,
            metadata_file=metadata_file,
            network=network,
            no_cache=no_cache,
            no_cache_filter=no_cache_filter,
            output=output_,
            platform=platform,
            progress=progress,
            pull=pull,
            push=push,
            quiet=logger.getEffectiveLevel() == logging.ERROR,
            secrets=secret,
            shm_size=shm_size,
            ssh=ssh,
            target=target,
            ulimit=ulimit,
        )
        if not exit_code:
            # Note that we have to duplicate the logic here
            # to ensure there is a docker image tag after a successful containerize.
            if not docker_image_tag:
                bento = _bento_store.get(bento_tag)
                docker_image_tag = (str(bento.tag),)
            # This section contains some duplicate logics from containerize_bento
            # to process log message.
            grpc_metrics_port = BentoMLContainer.grpc.metrics.port.get()
            logger.info(
                'Successfully built docker image for "%s" with tags "%s"',
                str(bento_tag),
                ",".join(docker_image_tag),
            )
            docker_tag = ",".join(docker_image_tag)
            if len(docker_image_tag) > 1:
                instruction = 'To run your newly built Bento container, use one of the above tags, and pass it to "docker run". %s'
            else:
                instruction = f'To run your newly built Bento container, pass "{docker_tag}" to "docker run". %s'
            instruction %= (
                'For example: "docker run -it --rm -p 3000:3000 %s serve --production".'
                % docker_tag
            )
            if enable_features is not None:
                # enable_features could be a tuple of comma-separated string.
                features = [
                    l for s in map(lambda x: x.split(","), enable_features) for l in s
                ]
                if "grpc" in features:
                    instruction += (
                        '\nAdditionally, to run your Bento container as a gRPC server, do: "docker run -it --rm -p 3000:3000 -p %s:%s %s serve-grpc --production"'
                        % (grpc_metrics_port, grpc_metrics_port, docker_tag)
                    )
            logger.info(instruction)
        sys.exit(exit_code)
