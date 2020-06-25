#!/usr/bin/env bash
set -e

BENTOML_SRC_DIR=$(git rev-parse --show-toplevel)

echo $BENTOML_SRC_DIR

cd "$BENTOML_SRC_DIR"

if ! [ -x "$(command -v bentoml)" ]; then
  echo "Installing BentoML dev dependencies"
  pip install -e .[dev]
fi

PROTO_PATH=$BENTOML_SRC_DIR/protos
PYOUT_PATH=$BENTOML_SRC_DIR/bentoml/yatai/proto

echo "Cleaning up existing proto generated py code.."
rm -rf "$PYOUT_PATH"
mkdir -p "$PYOUT_PATH"
touch "$PYOUT_PATH"/__init__.py

echo "Generate proto message code.."
find "$PROTO_PATH"/ -name '*.proto' | while read -r protofile; do
python -m grpc_tools.protoc \
  -I"$PROTO_PATH" \
  --python_out="$PYOUT_PATH" \
  "$protofile"
done

echo "Generate grpc service code.."
python -m grpc_tools.protoc \
  -I"$PROTO_PATH" \
  --python_out="$PYOUT_PATH" \
  --grpc_python_out="$PYOUT_PATH" \
  "$PROTO_PATH"/yatai_service.proto


echo "Fix imports in generated GRPC service code.."
find "$PYOUT_PATH"/ -name '*_pb2*.py' | while read -r pyfile; do
sed -i'.old' 's/^import \([^ ]*\)_pb2 \(.*\)$/import bentoml.yatai.proto.\1_pb2 \2/' "$pyfile"
sed -i'.old' 's/^from \([^ ]*\) import \([^ ]*\)_pb2\(.*\)$/from bentoml.yatai.proto.\1 import \2_pb2\3/' "$pyfile"
# Fix google.protobuf package imports
sed -i'.old' 's/^import bentoml.yatai.proto.google.\([^ ]*\)_pb2 as \([^ ]*\)$/import google.\1_pb2 as \2/' "$pyfile"
sed -i'.old' 's/^from bentoml.yatai.proto.google.\([^ ]*\) import \([^ ]*\)_pb2 as \([^ ]*\)$/from google.\1 import \2_pb2 as \3/' "$pyfile"
rm "$pyfile".old
done


PKG_PATH=$PYOUT_PATH
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

echo "Finish generating protobuf python code"

echo "Generate grpc code for javascript/typescript"
echo "Please make sure protobufjs is installed on your system"
echo "You can install with npm i -g protobufjs"

JS_GRPC_PATH=$BENTOML_SRC_DIR/bentoml/yatai/web/src/generated
echo "Cleaning up existing proto generated js code.."
rm -rf "$JS_GRPC_PATH"
mkdir -p "$JS_GRPC_PATH"

echo "Generating grpc JS code..."
pbjs -t static-module -w es6 --keep-case --force-number -o bentoml_grpc.js "$PROTO_PATH"/*.proto
pbts -o bentoml_grpc.d.ts bentoml_grpc.js

mv "$BENTOML_SRC_DIR"/bentoml_grpc.js "$JS_GRPC_PATH"
mv "$BENTOML_SRC_DIR"/bentoml_grpc.d.ts "$JS_GRPC_PATH"
echo "Finish generating protobuf js code"
