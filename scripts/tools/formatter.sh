#!/usr/bin/env bash

black --config ./pyproject.toml bentoml/ tests/ docker/ typings/

isort bentoml tests/ docker/ typings/
