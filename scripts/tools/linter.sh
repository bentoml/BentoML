#!/usr/bin/env bash

flake8 --config=setup.cfg bentoml tests docker

pylint --rcfile="./pyproject.toml" bentoml

pylint --rcfile="./pyproject.toml" --disable=E0401,F0010 tests docker
