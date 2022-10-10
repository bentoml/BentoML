#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
STUBS_GENERATOR="bentoml/stubs-generator"

cd "$GIT_ROOT/src" || exit 1

main() {
	local VERSION="${1:-v1alpha1}"
	# Use inline heredoc for even faster build
	# Keeping image as cache should be fine since we only want to generate the stubs.
	if [[ $(docker images --filter=reference="$STUBS_GENERATOR" -q) == "" ]] || test "$(git diff --name-only --diff-filter=d -- "$0")"; then
		docker buildx build --platform=linux/amd64 -t "$STUBS_GENERATOR" --load -f- . <<EOF
# syntax=docker/dockerfile:1.4-labs

FROM --platform=linux/amd64 python:3-slim-bullseye

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workspace

COPY <<-EOT /workspace/requirements.txt
    # Restrict maximum version due to breaking protobuf 4.21.0 changes
    # (see https://github.com/protocolbuffers/protobuf/issues/10051)
    # grpcio-tools==1.41.0 requires protobuf>=3.15.0, and 3.10 support are provided since protobuf>=3.19.0
    protobuf==3.19.4
    grpcio-tools==1.41
    mypy-protobuf>=3.3.0
EOT

RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -q -y --no-install-recommends --allow-remove-essential bash build-essential ca-certificates

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt \
	&& rm -rf /workspace/requirements.txt

EOF
	fi

	echo "Generating gRPC stubs..."
	find "bentoml/grpc/$VERSION" -type f -name "*.proto" -exec docker run --rm -it -v "$GIT_ROOT/src":/workspace --platform=linux/amd64 "$STUBS_GENERATOR" python -m grpc_tools.protoc -I. --grpc_python_out=. --python_out=. --mypy_out=. --mypy_grpc_out=. "{}" \;
}

if [ "${#}" -gt 1 ]; then
	echo "$0 takes one optional argument. Usage: $0 [v1alpha2]"
	exit 1
fi
main "$@"
