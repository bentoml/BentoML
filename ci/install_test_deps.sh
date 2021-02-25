#!/usr/bin/env bash
set -x

python -m pip install --upgrade pip
pip install .
pip install --upgrade --editable .[test]
