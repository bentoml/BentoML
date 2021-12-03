#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

need_cmd black || (echo "make sure to run \`make install-dev-deps\`"; exit 1);

INFO "(pyminify) reducing stubs size..."

for file in $(find typings/ -type f -iname '*.pyi'); do
  INFO "Processing $file..."
  pyminify --remove-literal-statements --no-remove-annotations --no-hoist-literals --no-rename-locals --no-remove-object-base --no-convert-posargs-to-args "$file"
  black --config "$GIT_ROOT"/pyproject.toml --pyi "$file"
  PASS "Finished processing $file..."
done
