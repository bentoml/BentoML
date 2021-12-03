#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

PROCESSED_TXT="$GIT_ROOT"/typings/processed.txt

if [[ ! -f "$PROCESSED_TXT" ]]; then
  touch "$PROCESSED_TXT"
fi

need_cmd black || (echo "make sure to run \`make install-dev-deps\`"; exit 1);

INFO "(pyminify) reducing stubs size..."

for file in $(find typings/ -type f -iname '*.pyi'); do
  if [ ! -z $(grep "$file" "$PROCESSED_TXT") ]; then
    PASS "$file already minified, skipping..."
    continue
  else
    INFO "Processing $file..."
    mv "$file" "$file".bak
    pyminify --remove-literal-statements --no-remove-annotations --no-hoist-literals --no-rename-locals --no-remove-object-base --no-convert-posargs-to-args "$file".bak &>> "$file"
    black --config "$GIT_ROOT"/pyproject.toml --pyi "$file"
    printf "%s\n" "$file" >> "$PROCESSED_TXT"
    /usr/bin/rm "$file".bak
    PASS "Finished processing $file..."
  fi
done
