#!/usr/bin/env python
import os
import subprocess
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
from git.repo import Repo

logger = logging.get_absl_logger()

SCRIPT_DIR = Path(__file__).parent
GIT_ROOT = SCRIPT_DIR.parent

R = Repo(GIT_ROOT)
G = R.git

FLAGS = flags.FLAGS

def call_cmd(cmd: str, shell: bool=True, check: bool=True):
    subprocess.run(cmd, shell=shell, check=check)


def main(argv: str) -> None:
    if len(argv) < 2:
        raise RuntimeError("Too few arguments. Positional `library` is required.")
    if len(argv) > 2:
        raise RuntimeError(f"Too much arguments. Only accepts one positional argument while got {len(argv)-1} positional arguments.")
    library = argv[1]

    # create stubs with pyright
    call_cmd(f"pyright --createstub {library}")
    # reducing stubs size and format stubs
    call_cmd(f". {Path(SCRIPT_DIR, 'tools', 'clean_stubs.sh').resolve()} {Path(GIT_ROOT, 'typings', library).resolve()}")
    # commit this change with signoff
    G.commit()

if __name__ == "__main__":
    app.run(main)
