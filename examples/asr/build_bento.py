from __future__ import annotations

import argparse
from os import path

import bentoml
from bentoml import Bento
from bentoml._internal.utils import resolve_user_filepath
from bentoml._internal.bento.build_config import BentoBuildConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", action="store_true", default=False)

    args = parser.parse_args()
    tag = "whispercpp_asr"

    try:
        bentos = bentoml.get(tag)
        if args.override:
            bentoml.delete(tag)
            raise bentoml.exceptions.NotFound("'--override', rebuilding bentos.")
        else:
            print(f"{bentos} already exists, use '--override' to rebuild.")
    except bentoml.exceptions.NotFound:
        bentofile = resolve_user_filepath("bentofile.yaml", None)

        with open(bentofile, "r", encoding="utf-8") as f:
            config = BentoBuildConfig.from_yaml(f)

        print(
            "Saved bento:",
            Bento.create(
                config,
                build_ctx=path.abspath(path.dirname(__file__)),
            ).save(),
        )
