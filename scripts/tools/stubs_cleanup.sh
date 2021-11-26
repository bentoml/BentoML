#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

source "$GIT_ROOT/scripts/ci/helpers.sh"

IGNORE=("$GITROOT/typings/numpy/typing/_ufunc.pyi" "$GITROOT/typings/numpy/typing/_callable.pyi")
PROCESSED=pyminifier/processed.txt

need_cmd git

if ! check_cmd pyminifier; then
  if [ ! -d "$GIT_ROOT/pyminifier" ]; then
    git clone https://github.com/liftoff/pyminifier.git;
  fi
  cd pyminifier
  pip install setuptools==57.0.0 && pip install -e .
  cd "$GIT_ROOT" || exit
fi

if [ ! -f "$PROCESSED" ]; then
  touch "$PROCESSED" || exit
fi

for file in $(find "$GIT_ROOT/typings" -name "*.pyi"); do
  if printf '%s\n' "${IGNORE[@]}" | grep -q '^'"$file"'$' || [ ! -z $(grep "$file" "$PROCESSED") ]; then
    INFO "skipping $file...."
    continue
  else
    INFO "processing $file ..."
    pyminifier -o "$file" "$file"
    sed -i 's/pass//' "$file"
    black --pyi "$file"
    printf "%s\n" "$file" >> "$PROCESSED"
  fi
done

INFO "Running black one more time just to make sure..."
black --pyi typings/**/*.pyi

# pip install -U setuptools
