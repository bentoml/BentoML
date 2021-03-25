#!/usr/bin/env bash
set -x

python -m pip install --upgrade pip
pip install urllib3>=1.25.10 six>=1.15
pip install .
pip install --upgrade --editable .[test]
