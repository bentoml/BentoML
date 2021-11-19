#!/usr/bin/env bash


fname=$(basename "$0")
dname=$(dirname "$0")

source "$dname/helpers.sh"

set_on_failed_callback "err=1"

GIT_ROOT=$(git rev-parse --show-toplevel)

err=0
use_poetry=0

PYTESTARGS=()
CONFIG_FILE="$dname/config.yml"
REQ_FILE="/tmp/rq.txt"

cd "$GIT_ROOT" || exit

yq_docker() {
  need_cmd docker
  docker run --rm -u "$(id -u)":"$(id -g)" -v "$GIT_ROOT":/bentoml bentoml/checker:1.0 yq "$@"
}

getval(){
  if check_cmd yq; then
    yq eval "$@" "$CONFIG_FILE";
  else
    yq_docker eval "$@" "$CONFIG_FILE"
  fi
}

run_python(){
  if [ "$use_poetry" -eq 1 ]; then
    need_cmd poetry
    poetry run python -m "$@"
  else
    python -m "$@"
  fi
}

validate_yaml() {
  # validate YAML file
  if ! [ -f "$CONFIG_FILE" ]; then
    FAIL "$CONFIG_FILE does not exists"
    exit 1
  fi

  if ! (yq_docker e --exit-status 'tag == "!!map" or tag== "!!seq"' "$CONFIG_FILE"> /dev/null); then
    FAIL "Invalid YAML file"
    exit 1
  fi
}


usage() {
  need_cmd cat

  cat <<HEREDOC
Running unit/integration tests with pytest and generate coverage reports. Make sure that given testcases is defined under $CONFIG_FILE.

Usage:
  $dname/$fname [-h|--help] [-v|--verbose] [--use-poetry] <target> <pytest_additional_arguments>

Flags:
  -h, --help            show this message
  -v, --verbose         set verbose scripts
  --use-poetry          use poetry to run scripts


If pytest_additional_arguments is given, this will be appended to given tests run.

Example:
  $ $dname/$fname pytorch --gpus
HEREDOC
  exit 2
}


parse_args() {
  for arg in "$@"; do
    case "$arg" in
      -h | --help)
        usage;;
      -v | --verbose)
        set -x;
        shift;;
      --use-poetry)
        need_cmd poetry;
        use_poetry=1;
        shift;;
      *)
        ;;
    esac
  done
  PYTESTARGS=( "${@:2}" )
  shift $(( OPTIND - 1 ))
}


parse_config() {
  target=$@
  test_dir=
  is_dir=
  override_name_or_path=
  external_scripts=
  type=

  test_dir=$(getval ".$target.root_test_dir")
  is_dir=$(getval ".$target.is_dir")
  override_name_or_path=$(getval ".$target.override_name_or_path")
  external_scripts=$(getval ".$target.external_scripts")
  type=$(getval ".$target.type")

  # processing file name
  if [[ "$override_name_or_path" != "" ]]; then
    fname="$override_name_or_path"
  elif [[ "$is_dir" == "false" ]]; then
    fname="test_""$target""_impl.py"
  elif [[ "$is_dir" == "true" ]]; then
    fname=""
  else
    fname="$target"
  fi

  # processing dependencies
  if check_cmd yq; then
    yq eval '.'"$target"'.dependencies[]' "$CONFIG_FILE" >"$REQ_FILE" || exit
  else
    yq_docker eval '.'"$target"'.dependencies[]' "$CONFIG_FILE" >"$REQ_FILE" || exit
  fi
}


main() {
  parse_args "$@"

  need_cmd make
  need_cmd curl
  need_cmd tr

  run_python pip install -U pip setuptools

  if ! docker info; then
    target_dir="$HOME/.local/bin"

    mkdir -p "$target_dir"
    echo "$target_dir" >> "$PATH"

    YQ_VERSION=4.14.2
    echo "Docker is not detected. Trying to install yq..."
    if ! check_cmd yq; then
      __shell=$(uname | tr '[:upper:]' '[:lower:]')
      YQ_BINARY=yq_"$__shell"_amd64
      curl -fsSLO https://github.com/mikefarah/yq/releases/download/v"$YQ_VERSION"/"$YQ_BINARY".tar.gz
      echo "tar $YQ_BINARY.tar.gz and move to /usr/bin/yq..."
      tar -zvxf "$YQ_BINARY.tar.gz" "$YQ_BINARY" && mv "$YQ_BINARY" "$target_dir"/yq
      rm -f ./"$YQ_BINARY".tar.gz
    else
      echo "Using yq via $(which yq)..."
    fi
  else
    make -f "$GIT_ROOT"/Makefile pull-checker-img
  fi

  for args in "$@"; do
    if [[ "$args" == "-"* ]]; then
      shift
    else
      argv="$args"
    fi
  done

  #  validate_yaml
  parse_config "$argv"

  OPTS=(--cov=bentoml --cov-config=.coveragerc --cov-report=xml:"$target.xml")

  if [ -n "$PYTESTARGS" ]; then
    OPTS=( "${OPTS[@]}" "${PYTESTARGS[@]}" )
  fi
  # setup tests environment
  if [ -f "$REQ_FILE" ]; then
    run_python pip install -r "$REQ_FILE" || exit 1
  fi

  if [ -n "$external_scripts" ]; then
    eval "$external_scripts" || exit 1
  fi

  if [ "$type" == 'e2e' ]; then
    cd "$GIT_ROOT"/"$test_dir"/"$fname" || exit 1
    path="."
  else
    path="$GIT_ROOT"/"$test_dir"/"$fname"
  fi

  if ! (run_python pytest "$path" "${OPTS[@]}"); then
    err=1
  fi

  # Return non-zero if pytest failed
  if ! test $err = 0; then
    FAIL "$type tests failed!"
    exit 1
  fi

  PASS "$type tests passed!"
}

main "$@" || exit 1

# vim: set ft=sh ts=2 sw=2 tw=0 et :
