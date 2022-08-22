#!/bin/sh

GIT_ROOT=$(git rev-parse --show-toplevel)

HOOKS_PATH="$GIT_ROOT/.git/hooks"

cd "$HOOKS_PATH" || exit

if [ ! -f "$HOOKS_PATH/commit-msg" ]; then
  ln -s "$GIT_ROOT/hooks/commit-msg" .
fi

if [ ! -f "$HOOKS_PATH/prepare-commit-msg" ]; then
  while true; do
      read -p "Do you want to setup sign-off commits? Make sure you know what you are doing :) " yn
    case $yn in
        [Yy]* )
          ln -s "$GIT_ROOT/hooks/prepare-commit-msg" .; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
  done
fi

if [ ! -f "$HOOKS_PATH/pre-commit" ]; then
  ln -s "$GIT_ROOT/hooks/pre-commit" .
fi
