#!/usr/bin/env bash
set -x

python -m pip install --upgrade pip
sudo python -m pip install --upgrade --editable ".[test]"
exit
