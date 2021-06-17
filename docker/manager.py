#!/usr/bin/env python3
#
#  ==========================================================================
#
#  Copyright (c) 2021 Atalaya Tech, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==========================================================================
#

from absl import flags
import yaml

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"

SPEC_MANIFEST = """
header: 

releases:
  model_server:
    devel:
      spec:
        - "{_PREFIX}{ubuntu2004-devel}"
        - "{_PREFIX}{centos8-devel}"
    runtime:
      spec:
        - "{_PREFIX}{ubuntu-2004}"
        - "{_PREFIX}{ubuntu-1804}"
        - "{_PREFIX}{centos8}"
        - "{_PREFIX}{amazonlinux2}"
        - "{_PREFIX}{alpine-3.14}"
    slim:
      spec:
        - "{_PREFIX}{debian10}"

components:
  ubuntu2004:
    - base_image: ubuntu:focal
    - add_to_name: ""
      args:
      partials:
    - add_to_name: "gpu"
      args:
      partials:
        -ubuntu/version
      cuda:
        - version: 11.3

"""

with open("manifest.yml", 'r') as spec_file:
    manifest = yaml.safe_load(spec_file)

print(manifest)