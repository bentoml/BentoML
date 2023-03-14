#!/bin/bash

set -ex

kaggle competitions download -c ieee-fraud-detection
rm -rf ./data/
unzip -o -d ./data/ ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
