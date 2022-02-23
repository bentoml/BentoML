#!/usr/bin/env python
import typing as t
import subprocess
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
from git.repo import Repo

logger = logging.get_absl_logger()

SCRIPT_DIR = Path(__file__).parent
GIT_ROOT = SCRIPT_DIR.parent

TYPINGS = "typings"
ACCEPTED_ACTIONS = ["create", "apply"]

G = Repo(GIT_ROOT).git

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "gpgsign",
    False,
    "Whether to sign all commits with gpg.",
    short_name="S",
)
flags.DEFINE_string(
    "branch",
    "main",
    "Default branch, set to `main`.",
    short_name="b",
)


def call_cmd(cmd: str, shell: bool = True, check: bool = False) -> None:
    subprocess.run(cmd, shell=shell, check=check)


def commit_msg(msg: str) -> t.List[str]:
    args = ["-sm", f'"{msg}"']
    if FLAGS.gpgsign:
        args = ["-S"] + args
    return args


def action_create(library: str) -> None:
    G.pull("--rebase", "origin", FLAGS.branch)
    branch = f"feat/__{library}_stubs__"
    if branch in G.branch("--all").split():
        G.branch("-D", branch)
    G.checkout("-b", branch)
    # create stubs with pyright
    call_cmd(f"pyright --createstub {library}")
    # reducing stubs size and format stubs
    call_cmd(
        f". {Path(SCRIPT_DIR, 'tools', 'clean_stubs.sh').resolve()} {Path(GIT_ROOT, TYPINGS, library).resolve()}"
    )
    # commit this change with signoff
    G.add(".")
    G.commit(*commit_msg(f"chore({TYPINGS}): add {library} stubs."))

    # format-patch
    with Path(GIT_ROOT, TYPINGS, f"stubs-{library}.patch").open(
        "w", encoding="utf-8"
    ) as stubs_file:
        stubs_file.write(G.format_patch("-k", "--stdout", "HEAD~1"))  # type: ignore
    call_cmd(f"rm -rf {Path(GIT_ROOT, TYPINGS, library).resolve()}")
    with Path(GIT_ROOT, ".gitignore").open("a+") as ignore_file:
        ignore_file.write(f"# added via scripts/typestubs.py DO NOT EDIT THE LINE BELOW #\ntypings/{library}\n")
    G.add(".")
    G.commit(
        *commit_msg(f"refactor({TYPINGS}): add stubs-{library}.patch.")
    )

    # rebase non-interactively
    # git reset --soft HEAD~2 then commit
    G.reset("--soft", "HEAD~2")
    G.commit(*commit_msg(f"feat({TYPINGS}): add stubs-{library}.patch"))
    G.push("origin", branch)


def action_apply(library: str) -> None:
    G.pull("--rebase", "origin", FLAGS.branch)
    # sanity check
    G.apply("--stat", Path(GIT_ROOT, TYPINGS, f"stubs-{library}.patch"))
    G.apply("--check", Path(GIT_ROOT, TYPINGS, f"stubs-{library}.patch"))

    # format-patch
    call_cmd(
        f"git am --signoff < {Path(GIT_ROOT, TYPINGS, f'stubs-{library}.patch').resolve()}"
    )


def main(argv: str) -> None:
    if len(argv) < 3:
        logger.error(
            "Too few arguments. Positional `actions` and `library` are required."
        )
    if len(argv) > 3:
        logger.error(
            f"Too much arguments. Only accepts two positional argument while got {len(argv)-1} positional arguments."
        )

    actions, library = argv[1], argv[2]
    if actions not in ACCEPTED_ACTIONS:
        logger.error(
            f"Unknown actions. Got {actions} while only accepts: {ACCEPTED_ACTIONS}"
        )
    if library == "":
        logger.error("`library` field is empty")

    if actions == "create":
        action_create(library)
    else:
        action_apply(library)


if __name__ == "__main__":
    app.run(main)
