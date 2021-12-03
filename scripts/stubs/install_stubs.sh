#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
DIRNAME=$GIT_ROOT/scripts/stubs
RM_BIN=/usr/bin/rm

source "$GIT_ROOT"/scripts/ci/helpers.sh

cd "$GIT_ROOT" || exit

IMPORTS="$DIRNAME"/imports.in
LOCK_FILE="$DIRNAME"/imports.lock
NUMPY_HASH_FILE="$DIRNAME"/numpy.lock

verlte() {
  [  "$1" = $(echo -e "$1\n$2" | sort -V | head -n1) ]
}

verlt() {
    [ "$1" = "$2" ] && return 1 || verlte $1 $2
}

numpy_stubs() {
  local NUMPY_STUBS_DIR="$GIT_ROOT/typings/numpy"

  mkdir -p "$NUMPY_STUBS_DIR" "$GIT_ROOT"/numpy
  if [ ! -f "$NUMPY_HASH_FILE" ]; then
    touch "$NUMPY_HASH_FILE"
  fi

  if [ ! -d /tmp/numpy ]; then
    git clone https://github.com/numpy/numpy.git /tmp/numpy
  fi

  cd /tmp/numpy/numpy
  local hash=$(git rev-parse HEAD)
  if [[ "$hash" != $(/usr/bin/cat "$NUMPY_HASH_FILE") ]]; then
    INFO "Processing numpy stubs from numpy/numpy..."
    echo "$hash" >| "$NUMPY_HASH_FILE"
    find . -name '*.pyi' -exec cp --parents '{}' "$GIT_ROOT"/numpy \;
    "$RM_BIN" -rf "$NUMPY_STUBS_DIR" && mv -f "$GIT_ROOT/numpy" "$NUMPY_STUBS_DIR"
  else
    INFO "numpy stubs is already up-to-date. Continue..."
  fi
}

get_library_version() {
  local libraries="$1"
  python <(cat <<EOF
from importlib_metadata import version
print(version("$libraries"))
EOF
)
}

libraries_stubs() {
  # libraries[0] would be imports, libraries[1] will be the package itself
  read -ra libraries <<<"$1"
  local needs_generate=0
  local imports="${libraries[0]}"
  local package="${libraries[1]}"
  local stubs_path="$GIT_ROOT"/typings/"$imports"

  local pypi_version=$(get_library_version "$package")
  local lock_version=$(grep "$package" "$LOCK_FILE" | sed 's/==/ /' | cut -d " " -f2)
  if verlt "$pypi_version" "$lock_version"; then
    FAIL "You need to upgrade $package before generating stubs..."
    FAIL "One can try \`pip install -U $package\`. If $package requires additional setups refers to Google :)"
    exit 1
  fi

  if ! grep -F "$package" "$LOCK_FILE" &>/dev/null || ! verlte "$pypi_version" "$lock_version"; then
    needs_generate=1
  fi
  if [ "$needs_generate" -eq 1 ]; then
    INFO "Processing $package stubs..."
    "$RM_BIN" -rf "$stubs_path"
    pyright --createstub "$imports"
    rpl="s/$package==$lock_version/$package==$pypi_version/g"
    grep -q "$package" "$LOCK_FILE" && sed -i "$rpl" "$LOCK_FILE" || echo "$package==$pypi_version" >>"$LOCK_FILE"
  else
    INFO "$package stubs already exists and up-to-date, skipping..."
  fi

}

main() {
  need_cmd black || (echo "Make sure to install dev-dependencies with `pip install -r $GIT_ROOT/requirements/dev-requirements.txt`"; exit 1);
  need_cmd pyright || (echo "You need to install pyright, https://github.com/microsoft/pyright"; exit 1);

  numpy_stubs && cd "$GIT_ROOT" || exit 1

  while read p; do
    libraries_stubs "$p"
  done <"$IMPORTS"

  if [ -d "$GIT_ROOT"/typings/multipart/multipart ]; then
    INFO "Cleaning up duplication..."
    "$RM_BIN" -rf "$GIT_ROOT"/typings/multipart/multipart
  fi

  INFO "Due to bash permission, make sure to run \`black --config $GIT_ROOT/pyproject.toml --pyi typings/**/*.pyi\` after the generating stubs."
}

main "$@"

