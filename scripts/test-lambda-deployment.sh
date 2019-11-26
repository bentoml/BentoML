#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

# Install deployment version of BentoML
cd GIT_ROOT
pip install -e .

# Generate a Bento Service and save it
python generate fake_service.py

# Create Lambda deployment and copy endpoint from previous result
bentoml --verbose deploy create soething

# make curl call to that endpoint
curl -i https://something.com

# Delete bentoml deployment

