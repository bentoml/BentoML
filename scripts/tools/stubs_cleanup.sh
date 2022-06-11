#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
MINIFY_OPTS=( --remove-literal-statements --no-remove-annotations --no-hoist-literals --no-rename-locals --no-remove-object-base --no-convert-posargs-to-args )

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

need_cmd "$EDITOR" || FAIL "You will need an editor to run this script, set your editor with env var: \$EDITOR=<your_text_editor>"
PROCESSED_TXT="$GIT_ROOT"/typings/processed.txt
: "${EDITOR:=/usr/bin/nano}"

if [[ ! -f "$PROCESSED_TXT" ]]; then
  touch "$PROCESSED_TXT"
fi

need_cmd black || (echo "black command not found, install dev dependencies with \`make install-dev-deps\`"; exit 1);

if [[ $(uname) == "Darwin" ]]; then
  SED_OPTS=( -E -i '' )
else
  SED_OPTS=( -E -i )
fi

INFO "(pyminify) reducing stubs size..."

for file in $(git ls-files | grep -e "**.pyi$"); do
  if [ ! -z $(grep "$file" "$PROCESSED_TXT") ]; then
    PASS "$file already minified, skipping..."
    continue
  else
    INFO "Processing $file ..."
    INFO "Removing pyright bugs..."
    sed "${SED_OPTS[@]}" "s/],:/]/g; s/,,/,/g; s/]\\\n    .../]: .../g" "$file"
    cp "$file" "$file".bak && rm "$file"
    if ! pyminify "${MINIFY_OPTS[@]}" "$file".bak > "$file"; then
      FAIL "Unable to processed $file, reverting to previous state. One can also use https://python-minifier.com/ to test where the problem may be. Make sure to match ${MINIFY_OPTS[@]}\nExitting now..."
      rm "$file"
      mv "$file".bak "$file"
      exit 1
    fi
    black --fast --config "$GIT_ROOT"/pyproject.toml --pyi "$file"
    printf "%s\n" "$file" >> "$PROCESSED_TXT"
    \rm "$file".bak
    PASS "Finished processing $file..."
  fi
done
