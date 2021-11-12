#!/usr/bin/env bash

mypy --config-file ./mypy.ini bentoml docker

pyright --stats
