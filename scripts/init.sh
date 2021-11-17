#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --top-level)

cd "$GIT_ROOT" || exit 1

source "$GIT_ROOT"/scripts/ci/helpers.sh

need_cmd curl

curl -sSL https://install.python-poetry.org | python -


cat <<HEREDOC
We will install BentoML with poetry to project virtualenv, then activate it via `poetry shell`.

You can then access `bentoml` CLI within the given virtualenv.

NOTE: If you already installed BentoMl from main with `pip install -e .`, uninstall `bentoml`
 in order to remove egg link within your Python installation. then proceed with installation using poetry.
HEREDOC

poetry install -vv -E "model-server"

poetry shell