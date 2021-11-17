#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source "$GIT_ROOT"/scripts/ci/helpers.sh

need_cmd curl

if ! check_cmd poetry; then
  curl -sSL https://install.python-poetry.org | python -
fi

cat <<HEREDOC
We will install BentoML with poetry to project virtualenv, then activate it via 'poetry shell'.

You can then access 'bentoml' CLI within the given virtualenv.

NOTE: If you already installed BentoMl from main with 'pip install -e .', uninstall 'bentoml'
 in order to remove egg link within your Python installation. then proceed with installation using poetry.
HEREDOC

poetry install -vv -E "model-server"

[[ "$VIRTUAL_ENV" == "" ]] && poetry shell
