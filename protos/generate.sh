#!/usr/bin/env bash
set -e

echo -e "\e[0;31m"
if [[ $EUID -eq 0 ]]; then
  cat <<ERROR
Currently running as ROOT. This could generate wrong permissions for generate gRPC files. Exiting...
ERROR
exit 1
fi
echo -e "\e[m"


if [[ -z "$BENTOML_REPO" ]]; then
  # Assuming running this script from anywhere within the BentoML git repository
  BENTOML_REPO=$(git rev-parse --show-toplevel)
fi

# YataiService protobuf
PROTO_PATH="$BENTOML_REPO"/protos
YATAI_PATH="$BENTOML_REPO"/bentoml/yatai
PY_OUT_PATH="$YATAI_PATH"/proto

# Test gRPC servicer
PROTO_TEST_PATH=$PROTO_PATH/tests
PY_TEST_OUT_PATH=$BENTOML_REPO/tests/yatai/proto

echo "Cleaning up existing proto generated py code.."
rm -rf "$PY_OUT_PATH" "$PY_TEST_OUT_PATH"
mkdir -p "$PY_OUT_PATH" "$PY_TEST_OUT_PATH"
touch "$PY_OUT_PATH"/__init__.py "$PY_TEST_OUT_PATH"/__init__.py "$PY_OUT_PATH"/__init__.pyi

echo "Generate proto message code.."
find "$PROTO_PATH"/ -name '*.proto' -not -path "${PROTO_TEST_PATH}/*" | while read -r protofile; do
python -m grpc_tools.protoc \
  -I"$PROTO_PATH" \
  --python_out="$PY_OUT_PATH" \
  --mypy_out="$PY_OUT_PATH" \
  "$protofile"
done

echo "Generate grpc test service code.."
python -m grpc_tools.protoc \
  -I"$PROTO_TEST_PATH" \
  --python_out="$PY_TEST_OUT_PATH" \
  --grpc_python_out="$PY_TEST_OUT_PATH" \
  "$PROTO_TEST_PATH"/mock_service.proto

echo "Generate grpc service code.."
python -m grpc_tools.protoc \
  -I"$PROTO_PATH" \
  --python_out="$PY_OUT_PATH" \
  --grpc_python_out="$PY_OUT_PATH" \
  --mypy_out="$PY_OUT_PATH" \
  "$PROTO_PATH"/yatai_service.proto

fix_grpc_service_code(){
  PKGNAME=$1
  OUT_DIR=$2

  echo "Fix imports for $PKGNAME in generated GRPC service code from $OUT_DIR ..."
  find "$OUT_DIR" -name '*_pb2*.py' | while read -r pyfile; do
  sed -i'.old' "s/^import \([^ ]*\)_pb2 \(.*\)$/import $PKGNAME.yatai.proto.\1_pb2 \2/" "$pyfile"
  sed -i'.old' "s/^from \([^ ]*\) import \([^ ]*\)_pb2\(.*\)$/from $PKGNAME.yatai.proto.\1 import \2_pb2\3/" "$pyfile"
  # Fix google.protobuf package imports
  sed -i'.old' "s/^import $PKGNAME.yatai.proto.google.\([^ ]*\)_pb2 as \([^ ]*\)$/import google.\1_pb2 as \2/" "$pyfile"
  sed -i'.old' "s/^from $PKGNAME.yatai.proto.google.\([^ ]*\) import \([^ ]*\)_pb2 as \([^ ]*\)$/from google.\1 import \2_pb2 as \3/" "$pyfile"
  rm "$pyfile".old
  done
}


fix_grpc_service_code "bentoml" "$PY_OUT_PATH"
fix_grpc_service_code "tests" "$PY_TEST_OUT_PATH"


edit_init() {
  PKG_PATH=$1

  PKGS=$(find "$PKG_PATH" -type d)
  for PKG in $PKGS; do
      SUBPACKAGES=$(find "$PKG" -maxdepth 1 -type d | grep -E -v "${PKG}$" | sort)
      MODULES=$(find "$PKG" -maxdepth 1 -iname "*.py" | grep -v "__init__.py" | sort)
      MODULE_COUNT=$(echo "$MODULES" | wc -w)
      PKG_INIT="${PKG}/__init__.py"
      echo "Writing Python package exports"
      echo "------------------------------"
      echo "PKG: ${PKG} (${PKG_INIT})"
      echo "SUBPACKAGES: ${SUBPACKAGES}"
      echo "FOUND MODULES: $MODULE_COUNT"
      # echo "MODULES: ${MODULES}"
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
  done

  echo "Done"
}

edit_init "$PY_OUT_PATH"
edit_init "$PY_TEST_OUT_PATH"

echo "Generate grpc code for javascript/typescript"
echo -e "\e[0;33m"
if ! command -v pbjs >/dev/null 2>&1; then
  cat <<WARN
WARNING: Make sure protobufjs is installed on your system. Use either npm or yarn to install dependencies:
    $ npm i -g protobufjs or
    $ yarn global add protobufjs
WARN
fi
echo -e "\e[m"

JS_OUT_PATH=$BENTOML_REPO/bentoml/yatai/web/src/generated
echo "Cleaning up existing proto generated js code.."
rm -rf "$JS_OUT_PATH"
mkdir -p "$JS_OUT_PATH"

echo "Generating gRPC JavaScript code..."
pbjs -t static-module -w es6 --keep-case --force-number -o bentoml_grpc.js "$PROTO_PATH"/*.proto
echo "Generating gRPC TypeScript code..."
pbts -o bentoml_grpc.d.ts bentoml_grpc.js

mv bentoml_grpc.js "$JS_OUT_PATH"
mv bentoml_grpc.d.ts "$JS_OUT_PATH"
echo "Done"