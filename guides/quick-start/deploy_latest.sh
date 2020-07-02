#!/usr/bin/env bash

latest_version=$(bentoml get -q IrisClassifier:latest | jq -r '.bentoServiceMetadata.version')
bentoml lambda deploy iris-classifier-dev -b IrisClassifier:"$latest_version"
