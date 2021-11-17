#!/usr/bin/env bash

mypy --config-file ./pyproject.toml bentoml/**/*.py
mypy --config-file ./pyproject.toml docker/**/*.py

pyright --stats
