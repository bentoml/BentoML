#!/usr/bin/env bash

set -eo pipefail

if [[ -z "$BENTOML_REPO" ]]; then
  # Assuming running this script from anywhere within the BentoML git repository
  BENTOML_REPO=$(git rev-parse --show-toplevel)
fi

# YataiService protobuf
PROTO_PATH="$BENTOML_REPO"/protos
PROTO_TEST_PATH="$PROTO_PATH"/tests
STUB_PATH="$BENTOML_REPO"/third_party/stubs

# Protos output for YataiServer
SERVER_OUT_PATH="$BENTOML_REPO"/yatai/yatai/proto
SERVER_STUB_PATH="$STUB_PATH"/yatai/yatai/proto

# Protos output for YataiClient
CLIENT_PATH=bentoml/_internal/yatai_client/proto
CLIENT_OUT_PATH="$BENTOML_REPO"/"$CLIENT_PATH"
CLIENT_STUB_PATH="$STUB_PATH"/"$CLIENT_PATH"

JS_OUT_PATH="$YATAI_SERVER_PATH"/web_server/src/generated

# Test gRPC servicer
PY_TEST_OUT_PATH="$BENTOML_REPO"/tests/unit/yatai/proto

run_check(){
  if [[ $EUID -eq 0 ]]; then
    log_error "Currently running as ROOT. This could generate wrong permissions for generate gRPC files. Exiting..."
    exit 1
  fi

  if ! command -v pbjs >/dev/null 2>&1; then
    log_warn - <<WARN
WARNING: Make sure protobufjs is installed on your system. Use either npm or yarn to install dependencies:
    $ npm i -g protobufjs or
    $ yarn global add protobufjs
WARN
  fi
}

cleanup(){
  log_info "Removing existing generated proto client code.."
  rm -rf "$SERVER_OUT_PATH" "$CLIENT_OUT_PATH" "$PY_TEST_OUT_PATH"
  rm -rf "$SERVER_STUB_PATH" "$CLIENT_STUB_PATH"
  rm -rf "$JS_OUT_PATH"

  mkdir -p "$SERVER_OUT_PATH" "$CLIENT_OUT_PATH" "$PY_TEST_OUT_PATH"
  mkdir -p "$SERVER_STUB_PATH" "$CLIENT_STUB_PATH"
  mkdir -p "$JS_OUT_PATH"

  touch "$PY_TEST_OUT_PATH"/__init__.py
  touch "$SERVER_OUT_PATH"/__init__.py "$SERVER_STUB_PATH"/__init__.pyi
  touch "$CLIENT_OUT_PATH"/__init__.py "$CLIENT_STUB_PATH"/__init__.pyi
}

generate_protos(){
  log_info "Generate proto message code.."
  find "$PROTO_PATH"/ -name '*.proto' -not -path "${PROTO_TEST_PATH}/*" | while read -r protofile; do
  python -m grpc_tools.protoc \
    -I"$PROTO_PATH" \
    --python_out="$SERVER_OUT_PATH" \
    --python_out="$CLIENT_OUT_PATH" \
    --mypy_out="$SERVER_STUB_PATH" \
    --mypy_out="$CLIENT_STUB_PATH" \
    "$protofile"
  done

  log_info "Generate gRPC test servicer.."
  python -m grpc_tools.protoc \
    -I"$PROTO_TEST_PATH" \
    --python_out="$PY_TEST_OUT_PATH" \
    --grpc_python_out="$PY_TEST_OUT_PATH" \
    "$PROTO_TEST_PATH"/mock_service.proto

  log_info "Generate yatai gRPC servicer.."
  python -m grpc_tools.protoc \
    -I"$PROTO_PATH" \
    --python_out="$SERVER_OUT_PATH" \
    --python_out="$CLIENT_OUT_PATH" \
    --grpc_python_out="$SERVER_OUT_PATH" \
    --grpc_python_out="$CLIENT_OUT_PATH" \
    --mypy_out="$SERVER_STUB_PATH" \
    --mypy_out="$CLIENT_STUB_PATH" \
    "$PROTO_PATH"/yatai_service.proto

  log_info "Generating gRPC JavaScript code..."
  pbjs -t static-module -w es6 --keep-case --force-number -o "$JS_OUT_PATH"/bentoml_grpc.js "$PROTO_PATH"/*.proto
  log_info "Generating gRPC TypeScript code..."
  pbts -o "$JS_OUT_PATH"/bentoml_grpc.d.ts "$JS_OUT_PATH"/bentoml_grpc.js
}

fix_grpc_service_code(){
  PKGNAME=$1
  OUT_DIR=$2

  log_info "Fix imports for $PKGNAME in generated gRPC service code from $OUT_DIR..."
  find "$OUT_DIR" -name '*_pb2*.py' | while read -r pyfile; do
  sed -i'.old' "s/^import \([^ ]*\)_pb2 \(.*\)$/import $PKGNAME.proto.\1_pb2 \2/" "$pyfile"
  sed -i'.old' "s/^from \([^ ]*\) import \([^ ]*\)_pb2\(.*\)$/from $PKGNAME.proto.\1 import \2_pb2\3/" "$pyfile"
  # Fix google.protobuf package imports
  sed -i'.old' "s/^import $PKGNAME.proto.google.\([^ ]*\)_pb2 as \([^ ]*\)$/import google.\1_pb2 as \2/" "$pyfile"
  sed -i'.old' "s/^from $PKGNAME.proto.google.\([^ ]*\) import \([^ ]*\)_pb2 as \([^ ]*\)$/from google.\1 import \2_pb2 as \3/" "$pyfile"
  rm "$pyfile".old
  done
}

edit_init() {
  PKG=$1

  MODULES=$(find "$PKG" -maxdepth 1 -iname "*.py" | grep -v "__init__.py" | sort)
  MODULE_COUNT=$(echo "$MODULES" | wc -w)
  PKG_INIT="${PKG}/__init__.py"
  log_info "Writing Python package exports"
  log_info "------------------------------"
  log_info "PKG: ${PKG} (${PKG_INIT})"
  log_info "SUBPACKAGES: ${SUBPACKAGES}"
  log_info "FOUND MODULES: ${MODULE_COUNT}"
  echo ""
  echo "__all__ = [" > "$PKG_INIT"
  for MODULE in $MODULES; do
      FILENAME=$(basename "$MODULE" .py)
      echo "    \"${FILENAME}\"," >> "$PKG_INIT"
  done
  for SUBPKG in $SUBPACKAGES; do
      SUBPKGNAME=$(basename "$SUBPKG")
      echo "    \"${SUBPKGNAME}\"," >> "$PKG_INIT"
  done
  echo "]" >> "$PKG_INIT"
}

log_warn() {
	echo -e "\033[2mWARN::\033[0m \e[1;32m$*\e[m" 1>&2
}

log_error() {
	echo -e "\033[2mERROR::\033[0m \e[1;31m$*\e[m" 1>&2
}

log_info() {
	echo -e "\033[2mINFO::\033[0m \e[1;34m$*\e[m" 1>&2
}

main(){
  run_check

  cleanup

  generate_protos

  fix_grpc_service_code "bentoml._internal.yatai_client" "$CLIENT_OUT_PATH"
  fix_grpc_service_code "yatai.yatai" "$SERVER_OUT_PATH"
  fix_grpc_service_code "tests.unit.yatai" "$PY_TEST_OUT_PATH"

  edit_init "$CLIENT_OUT_PATH"
  edit_init "$SERVER_OUT_PATH"
  edit_init "$PY_TEST_OUT_PATH"

  log_info "Finished generating gRPC servicer."
}

main