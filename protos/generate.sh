#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

# Install python grpcio-tools if it's not already installed
# pip install grpcio-tools

PROTO_PATH=$GIT_ROOT/protos
PYOUT_PATH=$GIT_ROOT/bentoml/proto

echo "Cleaning up existing proto generated py code.."
rm -f $PYOUT_PATH/*.py
rm -f $PYOUT_PATH/*.pyi
mkdir -p $PYOUT_PATH
touch $PYOUT_PATH/__init__.py

echo "Generate proto message code.."
find $PROTO_PATH/ -name '*.proto' | while read protofile; do
python -m grpc_tools.protoc \
  -I$PROTO_PATH \
  --python_out=$PYOUT_PATH \
  $protofile
done

echo "Generate grpc service code.."
python -m grpc_tools.protoc \
  -I$PROTO_PATH \
  --python_out=$PYOUT_PATH \
  --grpc_python_out=$PYOUT_PATH \
  $PROTO_PATH/yatai_service.proto


echo "Fix imports in generated GRPC service code.."
find $PYOUT_PATH/ -name '*_pb2*.py' | while read pyfile; do
sed -i '.old' 's/^import \([^ ]*\)_pb2 \(.*\)$/import bentoml.proto.\1_pb2 \2/' $pyfile
sed -i '.old' 's/^from \([^ ]*\) import \([^ ]*\)_pb2\(.*\)$/from bentoml.proto.\1 import \2_pb2\3/' $pyfile
# Fix google.protobuf package imports
sed -i '.old' 's/^import bentoml.proto.google.\([^ ]*\)_pb2 as \([^ ]*\)$/import google.\1_pb2 as \2/' $pyfile
sed -i '.old' 's/^from bentoml.proto.google.\([^ ]*\) import \([^ ]*\)_pb2 as \([^ ]*\)$/from google.\1 import \2_pb2 as \3/' $pyfile
rm $pyfile.old
done


PKG_PATH=$PYOUT_PATH
PKGS=$(find $PKG_PATH -type dir)
for PKG in $PKGS; do
    SUBPACKAGES=$(find $PKG -maxdepth 1 -type dir | egrep -v "${PKG}$" | sort)
    MODULES=$(find $PKG -maxdepth 1 -iname "*.py" | grep -v "__init__.py" | sort)
    MODULE_COUNT=$(echo $MODULES | wc -w)
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
