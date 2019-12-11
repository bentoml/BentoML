# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import tarfile
import logging
import tempfile
from contextlib import contextmanager
from urllib.parse import urlparse

import boto3
from bentoml.utils.s3 import is_s3_url
from bentoml.utils.usage_stats import track_load_finish, track_load_start
from bentoml.exceptions import BentoMLException
from bentoml.bundler.config import SavedBundleConfig


logger = logging.getLogger(__name__)


def _is_remote_path(bundle_path):
    return is_s3_url(bundle_path)


@contextmanager
def _resolve_remote_bundle_path(bundle_path):
    parsed_url = urlparse(bundle_path)
    bucket_name = parsed_url.netloc
    object_name = parsed_url.path.lstrip('/')

    s3 = boto3.client('s3')
    fileobj = io.BytesIO()
    s3.download_fileobj(bucket_name, object_name, fileobj)
    fileobj.seek(0, 0)

    with tarfile.open(mode="r:gz", fileobj=fileobj) as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = tar.getmembers()[0].name
            tar.extractall(path=tmpdir)
            yield os.path.join(tmpdir, filename)


def load_saved_bundle_config(bundle_path):
    if _is_remote_path(bundle_path):
        with _resolve_remote_bundle_path(bundle_path) as local_bundle_path:
            return load_saved_bundle_config(local_bundle_path)

    try:
        return SavedBundleConfig.load(os.path.join(bundle_path, "bentoml.yml"))
    except FileNotFoundError:
        raise BentoMLException(
            "BentoML can't locate config file 'bentoml.yml'"
            " in saved bundle in path: {}".format(bundle_path)
        )


def load_bento_service_metadata(bundle_path):
    return load_saved_bundle_config(bundle_path).get_bento_service_metadata_pb()


def load_bento_service_class(bundle_path):
    """
    Load a BentoService class from saved bundle in given path

    :param bundle_path: A path to Bento files generated from BentoService#save,
        #save_to_dir, or the path to pip installed BentoService directory
    :return: BentoService class
    """
    if _is_remote_path(bundle_path):
        with _resolve_remote_bundle_path(bundle_path) as local_bundle_path:
            return load_bento_service_class(local_bundle_path)

    config = load_saved_bundle_config(bundle_path)
    metadata = config["metadata"]

    # Load target module containing BentoService class from given path
    module_file_path = os.path.join(
        bundle_path, metadata["service_name"], metadata["module_file"]
    )
    if not os.path.isfile(module_file_path):
        # Try loading without service_name prefix, for loading from a installed PyPi
        module_file_path = os.path.join(bundle_path, metadata["module_file"])

    if not os.path.isfile(module_file_path):
        raise BentoMLException(
            "Can not locate module_file {} in saved bundle {}".format(
                metadata["module_file"], bundle_path
            )
        )

    # Prepend bundle_path to sys.path for loading extra python dependencies
    sys.path.insert(0, bundle_path)

    module_name = metadata["module_name"]
    if module_name in sys.modules:
        logger.warning(
            "Module `%s` already loaded, using existing imported module.", module_name
        )
        module = sys.modules[module_name]
    elif sys.version_info >= (3, 5):
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info >= (3, 3):
        from importlib.machinery import SourceFileLoader

        # pylint:disable=deprecated-method
        module = SourceFileLoader(module_name, module_file_path).load_module(
            module_name
        )
        # pylint:enable=deprecated-method
    else:
        raise BentoMLException("BentoML requires Python 3.4 and above")

    # Remove bundle_path from sys.path to avoid import naming conflicts
    sys.path.remove(bundle_path)

    model_service_class = module.__getattribute__(metadata["service_name"])
    # Set _bento_service_bundle_path, where BentoService will load its artifacts
    model_service_class._bento_service_bundle_path = bundle_path
    # Set cls._version, service instance can access it via svc.version
    model_service_class._bento_service_bundle_version = metadata["service_version"]

    return model_service_class


def load(bundle_path):
    """Load bento service from local file path or s3 path

    Args:
        bundle_path (str): The path that contains saved BentoService bundle,
            supporting both local file path and s3 path

    Returns:
        bentoml.service.BentoService: a loaded BentoService instance
    """

    if _is_remote_path(bundle_path):
        with _resolve_remote_bundle_path(bundle_path) as local_bundle_path:
            return load(local_bundle_path)

    track_load_start()

    svc_cls = load_bento_service_class(bundle_path)
    svc = svc_cls()

    track_load_finish(svc)
    return svc


def load_bento_service_api(bundle_path, api_name=None):
    if _is_remote_path(bundle_path):
        with _resolve_remote_bundle_path(bundle_path) as local_bundle_path:
            return load_bento_service_api(local_bundle_path, api_name)

    bento_service = load(bundle_path)
    return bento_service.get_service_api(api_name)
