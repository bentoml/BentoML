#!/usr/bin/env bash

fname=$(basename $0)

source ./scripts/ci/helpers.sh

err=0
set_on_failed_callback "err=1"

CONFIG_FILE="./scripts/.config/frameworks.yml"
GIT_ROOT=$(git rev-parse --show-toplevel)

num_args=${#@}
if [[ "$num_args" -ne 1 ]]; then
  cat <<ERROR
Usage: ./scripts/ci/"$fname" <frameworks>

Make sure to that given frameworks exists under "$CONFIG_FILE"

Example:
$ ./scripts/ci/"$fname" pytorch
ERROR
  exit 1
fi

yq_docker() {
  if [[ $(docker images --filter=reference='bentoml/checker' -q) == "" ]]; then
      docker pull bentoml/checker:1.0 || true
  fi
  docker run -i --rm -v "${PWD}":/bentoml bentoml/checker:1.0 yq "$@"
}

getval(){
  yq_docker eval "$@" "$CONFIG_FILE"
}

main() {
  # validate YAML file
  if ! (yq_docker e --exit-status 'tag == "!!map" or tag== "!!seq"' "$CONFIG_FILE"> /dev/null); then
    FAIL "Invalid YAML file"
  exit 1
  fi

  local framework=$1
  local test_dir is_dir

  test_dir=$(getval ".$framework.root_test_dir")
  is_dir=$(getval ".$framework.is_dir")

  # processing file name
  if [[ "$is_dir" == "false" ]]; then
    fname="test_""$framework""_impl.py"
  else
    fname="$framework"
  fi

  # processing dependencies
  yq_docker eval '.'"$framework"'.dependencies[]' "$CONFIG_FILE" >/tmp/rq.txt || exit

  cd "$GIT_ROOT" || exit
  pip install -r /tmp/rq.txt && rm /tmp/rq.txt
  pytest "$GIT_ROOT"/"$test_dir"/"$fname" --cov=bentoml --cov-config=.coveragerc --cov-report=xml:"$framework.xml"
  test $err = 0 # Return non-zero if pytest failed
}

main "$@"