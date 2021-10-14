#!/bin/sh

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

HOOKS_PATH="$GIT_ROOT/.git/hooks"

files=$(find "./dev/hooks"  -type f)
for f in $files; do
    fname=$(basename "$f")
    if [ ! -f "$HOOKS_PATH/$fname" ]; then
        ln -s "$GIT_ROOT/dev/hooks/$fname" "$HOOKS_PATH/$fname"
    fi
done
