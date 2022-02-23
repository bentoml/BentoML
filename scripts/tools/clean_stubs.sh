#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
MINIFY_OPTS=( --remove-literal-statements --no-remove-annotations --no-hoist-literals --no-rename-locals --no-remove-object-base --no-convert-posargs-to-args )

cd "$GIT_ROOT" || exit 1
source ./scripts/ci/helpers.sh

if [[ $(uname) == "Darwin" ]]; then
    SED_OPTS=( -E -i '' )
else
    SED_OPTS=( -E -i )
fi

process_file(){
    FILE="$1"
    INFO "(pyminify) Processing and reducing size for $FILE ... 🤖"
    INFO "(sed) Removing pyright bugs..."
    sed "${SED_OPTS[@]}" "s/],:/]/g; s/,,/,/g; s/]\\\n    .../]: .../g" "$FILE"
    cp "$file" "$file".bak && rm "$FILE"
    if ! pyminify "${MINIFY_OPTS[@]}" "$FILE".bak > "$FILE"; then
        FAIL "Unable to processed $FILE, reverting to previous state. One can also use https://python-minifier.com/ to test where the problem may be. Make sure to match ${MINIFY_OPTS[@]}\nExitting now..."
        rm "$FILE"
        mv "$FILE".bak "$FILE"
        exit 1
    fi
    black --fast --config "$GIT_ROOT"/pyproject.toml --pyi --quiet "$FILE"
    isort --quiet .
    \rm "$file".bak
    PASS "Finished processing $FILE... 🥰"
}

main() {
    need_cmd black || (echo "black command not found, install dev dependencies with \`make install-dev-deps\`"; exit 1);
    DIRECTORY="$1"

    for file in $(find "$DIRECTORY" -type f -name '*.pyi'); do
        process_file "$file"
    done
}

main "$@"
