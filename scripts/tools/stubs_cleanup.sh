#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
MINIFY_OPTS=( --remove-literal-statements --no-remove-annotations --no-hoist-literals --no-rename-locals --no-remove-object-base --no-convert-posargs-to-args )

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

need_cmd vim || FAIL "You will need to install vim to use this script, or use any editor of choice\nrequires to set \$EDITOR=<your_text_editor>"
PROCESSED_TXT="$GIT_ROOT"/typings/processed.txt
EDITOR=$(echo "$EDITOR") || /usr/bin/vim

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
    INFO "Removing pyright bugs..."
    sed -i "s/],:/]/g" "$file"
    sed -i "s/,,/,/g" "$file"
    sed -i "s/]\n    .../]: .../g" "$file"
    # sed -i "s/]$/]:/g" "$file"
    cp "$file" "$file".bak && rm "$file"
    if ! pyminify "${MINIFY_OPTS[@]}" "$file".bak &> "$file"; then
      FAIL "unable to processed $file, reverting to previous state, opening editor to fix..."
      rm "$file"
      mv "$file".bak "$file"
      "$EDITOR" "$file" || exit
      cp "$file" "$file".bak
      if ! pyminify "${MINIFY_OPTS[@]}" "$file".bak &> "$file"; then
        FAIL "Failed to fix $files. One can also use https://python-minifier.com/ to test where the problem may be. Make sure to match ${MINIFY_OPTS[@]}\nExitting now..."
        rm "$file"
        mv "$file".bak "$file"
        exit 1
      fi
    fi
    black --fast --config "$GIT_ROOT"/pyproject.toml --pyi "$file"
    printf "%s\n" "$file" >> "$PROCESSED_TXT"
    /usr/bin/rm "$file".bak
    PASS "Finished processing $file..."
  fi
done
