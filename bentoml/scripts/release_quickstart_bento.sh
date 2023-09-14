#!/bin/bash

if [ "$#" -eq 1 ]; then
	BENTOML_VERSION=$1
else
	BENTOML_VERSION=$(python -c "import bentoml; print(bentoml.__version__)")
	echo "Releasing with current BentoML Version $BENTOML_VERSION"
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

docker run --platform=linux/amd64 \
	-v $GIT_ROOT:/bentoml \
	-v $HOME/.aws:/root/.aws \
	python:3.8-slim /bin/bash -c """\
pip install -U pip
pip install "bentoml[grpc]==$BENTOML_VERSION"
cd /bentoml/examples/quickstart
pip install -r ./requirements.txt
python train.py
bentoml build
pip install fs-s3fs
bentoml export iris_classifier:latest s3://bentoml.com/quickstart/iris_classifier.bento
"""
