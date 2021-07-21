#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
STUB_DIR="$GIT_ROOT"/third_party/stubs
PROTO_PATH="$GIT_ROOT"/protos
PROTO_TEST_PATH="$PROTO_PATH"/tests

# gRPC stub path
SERVER_STUB_PATH="$STUB_DIR"/yatai/proto
CLIENT_STUB_PATH="$STUB_DIR"/bentoml/_internal/yatai_client/proto

cd "$GIT_ROOT" || exit 1

mkdir -p "$STUB_DIR"

log_info(){
	echo -e "\033[2mINFO::\033[0m \e[1;34m$*\e[m" 1>&2
}

gen_protos_stub(){
  OUT_PATH=$1
  log_info "Generate proto stub to $OUT_PATH..."
  find "$PROTO_PATH"/ -name '*.proto' -not -path "${PROTO_TEST_PATH}/*" | while read -r protofile; do
    python -m grpc_tools.protoc \
      -I"$PROTO_PATH" \
      --mypy_out="$OUT_PATH" \
      "$protofile"
  done

  log_info "Generate yatai servicer stub to $OUT_PATH..."
  python -m grpc_tools.protoc \
    -I"$PROTO_PATH" \
    --mypy_out="$OUT_PATH" \
    "$PROTO_PATH"/yatai_service.proto
}

fix(){
  log_info "Generate stubs for bentoml..."
  stubgen -p bentoml -o "$STUB_DIR" -v

  log_info "Generate stubs for yatai..."
  cd yatai && stubgen -p yatai -o "$STUB_DIR" -v

  log_info "Remove unnecessary stub..."
  find "$STUB_DIR" -type f -iname "_version.pyi" -exec sh -c '
    file=$1
    rm "$file"
  ' {} {} \;

  rm -rf "$(find "$STUB_DIR" -type d -iname "migrations")" "$SERVER_STUB_PATH" "$CLIENT_STUB_PATH"

  mkdir -p "$SERVER_STUB_PATH" "$CLIENT_STUB_PATH"
  touch  "$CLIENT_STUB_PATH"/__init__.pyi "$SERVER_STUB_PATH"/__init__.pyi
}


main(){
  fix

  gen_protos_stub "$SERVER_STUB_PATH"
  gen_protos_stub "$CLIENT_STUB_PATH"
}

main