#!/bin/bash

head --lines=200 ./data/test_transaction.csv | curl -X POST -H 'Content-Type: text/csv' --data-binary @- http://0.0.0.0:3000/is_fraud
