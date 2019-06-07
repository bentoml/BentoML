# Deploy BentoML bundle to Clipper cluster


## Overview 

Clipper is a low-latency prediction serving system for machine learning.  We
will train a basic Iris classifier with SKlearn, save it as BentoML bundle and
deploy it to Clipper cluster.

## Prerequisites

* An activated python 3.6 environment.
* Docker. You can install it on your system with this [Instructions](https://docs.docker.com/install/)


## Running example

1. Install pypi packages
```python
pip install -r requirements.txt
```

2. Run the example
```python
python deploy-clipper.py
```
