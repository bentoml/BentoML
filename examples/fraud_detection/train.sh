#!/bin/bash

jupyter nbconvert --to notebook --inplace --execute ./IEEE-CIS-Fraud-Detection.ipynb --debug 2>&1 | grep -v '^\[NbConvertApp\]' 
