#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

# make sure to kill the python server when existing the script
trap 'kill $(jobs -p)' EXIT

echo "Initial docs build..."
sphinx-build $GIT_ROOT/docs $GIT_ROOT/built-docs

echo "Starting local http server for preview..."
python3 -m http.server --directory $GIT_ROOT/built-docs &

echo "Open browser..."
open -a "Google Chrome" http://0.0.0.0:8000/

fswatch -o $GIT_ROOT/docs $GIT_ROOT/bentoml | while read; \
  do \
    echo "Change detected, rebuilding docs..."
    sphinx-build -E $GIT_ROOT/docs $GIT_ROOT/built-docs

    # refresh page
    osascript -e '
set local_http_server_url to "http://0.0.0.0:8000/"
tell application "Google Chrome"
    repeat with chrome_window in windows
        repeat with t in tabs of chrome_window
            if URL of t starts with local_http_server_url then
                tell t to reload
                return
            end if
        end repeat
    end repeat
    open location local_http_server_url
    activate
end tell
    '
  done
