from typing import Optional, List, Dict

from bentoml.configuration.containers import BentoMLContainer


SUPPORTED_PYTHON_VERSION: List = ['3.7', '3.8']
SUPPORTED_BASE_OSES: List = ['slim', 'centos7', 'centos8']

RESERVED_GPU_OSES: List = SUPPORTED_BASE_OSES

# NOTES: model-server only.
SUPPORTED_RELEASES_COMBINATION: Dict[str, List[str]] = {
    'cudnn': SUPPORTED_BASE_OSES,
    'devel': SUPPORTED_BASE_OSES,
    'runtime': SUPPORTED_BASE_OSES + ['ami2', 'alpine3.14'],
}


class ProvidedImages(object):
    def __init__(self, os: str, python_version: str, gpu: Optional[bool] = False):
        self._os = os
        self._python_version = python_version
        self._gpu = gpu
        self._images = {}

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return self._images

    def _tag_fmt(self):
        pass