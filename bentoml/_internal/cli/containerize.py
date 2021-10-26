import click

from ..utils.docker_utils import validate_tag


def add_containerize_command(cli):
    @cli.command(
        help="Containerizes given Bento into a ready-to-use Docker image.",
    )
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-t",
        "--tag",
        help="Optional image tag. If not specified, Bento will generate one from "
        "the name of the Bento.",
        required=False,
        callback=validate_tag,
    )
    @click.option(
        "--build-arg", multiple=True, help="pass through docker image build arguments"
    )
    def containerize(bento_tag, tag, build_arg):
        """Containerize specified Bento.

        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml containerize` command also supports the use of the `latest` tag
        which will automatically use the last built version of your Bento.

        You can provide a tag for the image built by Bento using the
        `--tag` flag. Additionally, you can provide a `--push` flag,
        which will push the built image to the Docker repository specified by the
        image tag.

        You can also prefixing the tag with a hostname for the repository you wish
        to push to.
        e.g. `bentoml containerize IrisClassifier:latest --push --tag
        repo-address.com:username/iris` would build a Docker image called
        `username/iris:latest` and push that to docker repository at repo-address.com.

        By default, the `containerize` command will use the current credentials
        provided by Docker daemon.
        """
        pass
