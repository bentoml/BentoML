#!/usr/bin/env bash
set -x

apt-get install libenchant

python -m pip install --upgrade pip
pip install .
pip install --upgrade .[doc_builder]
