#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
STUB_DIR="$GIT_ROOT"/third_party/stubs

cd "$GIT_ROOT" || exit 1

stubgen -p bentoml -o "$STUB_DIR" -v

cd yatai && stubgen -p yatai -o "$STUB_DIR" -v

find "$STUB_DIR" -type f -iname "_version.pyi" -exec sh -c '
  file=$1
  rm "$file"
' {} {} \;

rm -rf "$(find "$STUB_DIR" -type d -iname "migrations")"