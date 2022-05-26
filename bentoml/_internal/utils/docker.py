from __future__ import annotations

import os
import re
import logging
from typing import TYPE_CHECKING

from rich.syntax import Syntax

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from click import Context
    from click import Parameter


def to_valid_docker_image_name(name: str) -> str:
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return name.lower().strip("._-")


def to_valid_docker_image_version(version: str) -> str:
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return version.encode("ascii", errors="ignore").decode().lstrip(".-")[:128]


def validate_tag(_: Context, __: Parameter, tag: str | None) -> str | None:
    if tag is None:
        return tag

    if ":" in tag:
        name, version = tag.split(":")[:2]
    else:
        name, version = tag, None

    valid_name_pattern = re.compile(
        r"""
        ^(
        [a-z0-9]+      # alphanumeric
        (.|_{1,2}|-+)? # seperators
        )*$
        """,
        re.VERBOSE,
    )
    valid_version_pattern = re.compile(
        r"""
        ^
        [a-zA-Z0-9] # cant start with .-
        [ -~]{,127} # ascii match rest, cap at 128
        $
        """,
        re.VERBOSE,
    )

    if not valid_name_pattern.match(name):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. "
            "Name components may contain lowercase letters, digits "
            "and separators. A separator is defined as a period, "
            "one or two underscores, or one or more dashes."
        )
    if version and not valid_version_pattern.match(version):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. "
            "A tag name must be valid ASCII and may contain "
            "lowercase and uppercase letters, digits, underscores, "
            "periods and dashes. A tag name may not start with a period "
            "or a dash and may contain a maximum of 128 characters."
        )
    return tag


FROM_WARNING = """\
[bold yellow]FROM[/] instruction is [bold red]NOT SUPPORTED[/bold red] in `dockerfile_template`. 
BentoML will auto-generate a `FROM` instruction that extends the generated Bento dockerfile.

If you want to use a custom `FROM` instruction, we recommend you to build your own custom Docker container, then 
use `base_image` under `bentofile.yaml`.
        """
ENTRYPOINT_WARNING = """\
Since only the [bold yellow link=https://docs.docker.com/engine/reference/builder/#entrypoint]last `ENTRYPOINT` instruction[/link bold yellow] in the Dockerfile will have an effect, the given Bento container will fail when one uses: `docker run -p 3000:3000 bento:tag`.
To maintain the default behaviour of the Bento container and keep the new `ENTRYPOINT` instruction, update the `ENTRYPOINT` to the following (given default `BENTO_PATH`={}):

    [red]- {}[/red]
    [green]+ {}[/green]
"""
CMD_WARNING = """\
        """
WORKDIR_WARNING = """\
        """


WARNINGS = {
    "FROM": FROM_WARNING,
    "ENTRYPOINT": ENTRYPOINT_WARNING,
    "CMD": CMD_WARNING,
    "WORKDIR": WORKDIR_WARNING,
}


def pretty_warning_logs(parsed_command: str, instruction_warning: str) -> None:
    m = instruction_regex.match(parsed_command)
    if m is not None:
        instruction = m.groups()[0]
        pretty_cmd = Syntax(parsed_command, "dockerfile")
        logger.warning(
            f"""\
    [bold yellow]{instruction}[/] instruction is given in `dockerfile_template`:

        {pretty_cmd}

    {instruction_warning}
                """
        )


WARNING_MESSAGE = """\
[bold yellow]NOTE: Since [bold red]`dockerfile_templates`[/] is provided:
    1. `dockerfile_templates` shouldn't contain any [bold red]`FROM`[/] instruction. BentoML will generate one for you.
        An example `custom.Dockerfile` provided via `docker.dockerfile_templates=/path/to/custom.Dockerfile`:

            COPY ./README.md .

            RUN python -c "import sys; print('Python version: {'{}'}'.format(sys.version))"

    2. During [bold magenta]{tag}[/]'s dockerfile generation, BentoML changes `WORKDIR` to `/home/bentoml/bento` for the Docker container.
        For any COPY, ADD instruction, make sure that the copying file is available under {build_ctx}.

    3. The Dockerfile for building `{tag}` contains:

            ENTRYPOINT [ "./env/docker/entrypoint.sh" ]
            CMD ["bentoml", "serve", ".", "--production"]

        where the `./env/docker/entrypoint.sh` sets up the environment correctly to run bentoml in the subsequent `CMD`.
        This allows us to run the container directly as an executable: `docker run -p 3000:3000 {tag}`.

        If the occassion arises where inside given `dockerfile_templates` contains a custom `ENTRYPOINT`, or changing `WORKDIR`:

            ENTRYPOINT [ "my_command" ]

            WORKDIR "/my/custom/workdir"

        Only the last `ENTRYPOINT` instruction in the Dockerfile will have an effect. In order to maintain the default bento container behaviour, add  `/home/bentoml/bento/env/docker/entrypoint.sh` as first elements of `ENTRYPOINT`:

            ENTRYPOINT [ "/home/bentoml/bento/env/docker/entrypoint.sh", "custom_scripts.sh" ]
            CMD ["bentoml", "serve", "/home/bentoml/bento", "--production"]

        Refers to [bold yellow][link=https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact]ENTRYPOINT and CMDs interaction[/link][/bold yellow] for further details.
                """


def _strike(text: str) -> str:
    return "".join(["\u0336{}".format(c) for c in text])


def process_entrypoint_cmd(instruction: str) -> str:
    ...


def rstrip_eol(text: str, continuation_char: str = "\\") -> str:
    text = text.rstrip()
    if text.endswith(continuation_char):
        text = text[:-1]
    return text.strip()


def clean_comments(instruction: str):
    instruction = re.sub(r"^\s*#\s*", "", instruction)
    return re.sub(r"\n", "", instruction)


# https://docs.docker.com/engine/reference/builder/#parser-directives
# escape directive regex: # escape=`
escape_directive_regex = re.compile(r"^\s*#\s*escape\s*=\s*(\\|`)\s*$", re.I)
# syntax directive regex: # syntax=dockerfile
syntax_directive_regex = re.compile(r"^\s*#\s*syntax\s*=\s*(.*)\s*$", re.I)

# instruction regex
instruction_regex = re.compile(r"^\s*(\S+)\s+(.*)$")
# continuation line regex
continuation_regex = re.compile(r"^.*\\\s*$")


def validate_dockerfile_template(path_or_instruction: str):
    if os.path.exists(path_or_instruction):
        with open(path_or_instruction, "r", encoding="utf-8") as f:
            instructions = list(filter(lambda x: x != "\n", f.readlines()))
    else:
        instructions = path_or_instruction

    # filter out comments
    instructions = list(filter(lambda x: not re.match(r"^\s*#", x), instructions))

    for line in instructions:
        if escape_directive_regex.match(line) or syntax_directive_regex.match(line):
            raise BentoMLException(
                """\
Directives are not supported in `dockerfile_template`. This includes `# escape=` and `# syntax=` directives.
Generated Dockerfile is already using syntax directive to take advantage of BuildKit frontend.
Refers to https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md for more details on how to use BuildKit frontend.
                    """
            )
        # INSTRUCTION regex
        if not instruction_regex.match(line):
            continue
