#!/usr/bin/env bash

LIBRARIES=(pandas yaml xgboost transformers transformers starlette PIL huggingface_hub)
GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

target_np_stubs="$GIT_ROOT/typings/numpy"

mkdir -p "$target_np_stubs"

if [ ! -d /tmp/numpy ]; then
  git clone https://github.com/numpy/numpy.git /tmp/numpy
fi

mkdir -p "$GIT_ROOT/numpystubs"
cd /tmp/numpy/numpy && find . -name '*.pyi' -exec cp --parents '{}' "$GIT_ROOT/numpystubs" \;

if [ -d "$target_np_stubs" ]; then
  /usr/bin/rm -rf "$target_np_stubs"
fi

mv -f "$GIT_ROOT/numpystubs" "$target_np_stubs"

for lib in "$LIBRARIES[@]"; do
  pyright --createstubs "$lib"
done
