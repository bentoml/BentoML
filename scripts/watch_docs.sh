#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)

# make sure to kill the python server when existing the script
trap 'kill $(jobs -p)' EXIT

echo "removing previous build and make a new one.."
cd "$GIT_ROOT"/docs
make clean && make html



echo "Starting local http server for preview..."
python3 -m http.server --directory "$GIT_ROOT"/docs/build/html &

echo "Open browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  fswatch -o "$GIT_ROOT/docs/source" "$GIT_ROOT/bentoml" | while read -r; do
    echo "Change detected, rebuilding docs..."
    cd "$GIT_ROOT"/docs
    make clean && make html

    # refresh page
    osascript -e '
set local_http_server_url to "http://0.0.0.0:8000/"
tell application "Google Chrome"
    set found to false
    set tab_index to -1
    repeat with chrome_window in every window
        set tab_index to 0
        repeat with chrome_tab in every tab of chrome_window
            set tab_index to tab_index + 1
            if URL of chrome_tab start with local_http_server_url then
                set found to true
                exit
            end if
        end repeat
        if found then
          exit repeat
        end if
    end repeat

    if found then
      tell chrome_tab to reload
      set active tab index of chrome_window to tab_index
      set index of chrome_window to 1
    else
      tell window 1 to make new tab with properties {URL: local_http_server_url}
    end if
end tell
    '
  done
else
  xdg-open http://localhost:8000
  while inotifywait -e modify -r "$GIT_ROOT"/docs/source "$GIT_ROOT"/bentoml; do
    echo "Change detected, rebuilding docs..."
    cd "$GIT_ROOT"/docs && make clean && make html
  done
fi
