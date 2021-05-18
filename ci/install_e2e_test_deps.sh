#!/usr/bin/env bash
set -x

python -m pip install --upgrade pip
sudo python -m pip install urllib3>=1.25.10 six>=1.15 psycopg2
sudo python -m pip install --upgrade --editable ".[test]"
exit
