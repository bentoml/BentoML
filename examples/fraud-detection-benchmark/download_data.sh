#!/bin/bash

kaggle competitions download -c ieee-fraud-detection
unzip -o -d ./data/ ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
