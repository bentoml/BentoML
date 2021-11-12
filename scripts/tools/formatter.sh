#!/usr/bin/env bash

black --config ./pyproject.toml bentoml/ tests/ docker/

isort bentoml tests/ docker/
