#!/usr/bin/env bash

flake8 --config=setup.cfg bentoml tests docker

pylint --rcfile="./pylintrc" bentoml

pylint --rcfile="./pylintrc" --disable=E0401,F0010 tests docker
