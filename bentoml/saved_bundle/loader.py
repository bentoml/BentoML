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

import io
import os
import sys
import tarfile
import logging
import tempfile
import shutil
from functools import wraps
from contextlib import contextmanager
from urllib.parse import urlparse
from typing import TYPE_CHECKING
from pathlib import PureWindowsPath, PurePosixPath

from bentoml.utils.s3 import is_s3_url
from bentoml.utils.gcs import is_gcs_url
from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle.config import SavedBundleConfig
from bentoml.saved_bundle.pip_pkg import ZIPIMPORT_DIR

if TYPE_CHECKING:
    from bentoml.yatai.proto.repository_pb2 import BentoServiceMetadata

logger = logging.getLogger(__name__)


def _is_http_url(bundle_path) -> bool:
    try:
        return urlparse(bundle_path).scheme in ["http", "https"]
    except ValueError:
        return False


def _is_remote_path(bundle_path) -> bool:
    return isinstance(bundle_path, str) and (
        is_s3_url(bundle_path) or is_gcs_url(bundle_path) or _is_http_url(bundle_path)
    )


@contextmanager
def _resolve_remote_bundle_path(bundle_path):
    if is_s3_url(bundle_path):
        import boto3

        parsed_url = urlparse(bundle_path)
        bucket_name = parsed_url.netloc
        object_name = parsed_url.path.lstrip('/')

        s3 = boto3.client('s3')
        fileobj = io.BytesIO()
        s3.download_fileobj(bucket_name, object_name, fileobj)
        fileobj.seek(0, 0)
    elif is_gcs_url(bundle_path):
        try:
            from google.cloud import storage
        except ImportError:
            raise BentoMLException(
                '"google-cloud-storage" package is required. You can install it with '
                'pip: "pip install google-cloud-storage"'
            )

        gcs = storage.Client()
        fileobj = io.BytesIO()
        gcs.download_blob_to_file(bundle_path, fileobj)
        fileobj.seek(0, 0)
    elif _is_http_url(bundle_path):
        import requests

        response = requests.get(bundle_path)
        if response.status_code != 200:
            raise BentoMLException(
                f"Error retrieving BentoService bundle. "
                f"{response.status_code}: {response.text}"
            )
        fileobj = io.BytesIO()
        fileobj.write(response.content)
        fileobj.seek(0, 0)
    else:
        raise BentoMLException(f"Saved bundle path: '{bundle_path}' is not supported")

    with tarfile.open(mode="r:gz", fileobj=fileobj) as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = tar.getmembers()[0].name
            tar.extractall(path=tmpdir)
            yield os.path.join(tmpdir, filename)


def resolve_remote_bundle(func):
    """Decorate a function to handle remote bundles."""

    @wraps(func)
    def wrapper(bundle_path, *args):
        if _is_remote_path(bundle_path):
            with _resolve_remote_bundle_path(bundle_path) as local_bundle_path:
                return func(local_bundle_path, *args)

        return func(bundle_path, *args)

    return wrapper


@resolve_remote_bundle
def load_saved_bundle_config(bundle_path) -> "SavedBundleConfig":
    try:
        return SavedBundleConfig.load(os.path.join(bundle_path, "bentoml.yml"))
    except FileNotFoundError:
        raise BentoMLException(
            "BentoML can't locate config file 'bentoml.yml'"
            " in saved bundle in path: {}".format(bundle_path)
        )


def load_bento_service_metadata(bundle_path: str) -> "BentoServiceMetadata":
    return load_saved_bundle_config(bundle_path).get_bento_service_metadata_pb()


def _find_module_file(bundle_path, service_name, module_file):
    # Simply join full path when module_file is just a file name,
    # e.g. module_file=="iris_classifier.py"
    module_file_path = os.path.join(bundle_path, service_name, module_file)
    if not os.path.isfile(module_file_path):
        # Try loading without service_name prefix, for loading from a installed PyPi
        module_file_path = os.path.join(bundle_path, module_file)

    # When module_file is located in sub directory
    # e.g. module_file=="foo/bar/iris_classifier.py"
    # This needs to handle the path differences between posix and windows platform:
    if not os.path.isfile(module_file_path):
        if sys.platform == "win32":
            # Try load a saved bundle created from posix platform on windows
            module_file_path = os.path.join(
                bundle_path, service_name, str(PurePosixPath(module_file))
            )
            if not os.path.isfile(module_file_path):
                module_file_path = os.path.join(
                    bundle_path, str(PurePosixPath(module_file))
                )
        else:
            # Try load a saved bundle created from windows platform on posix
            module_file_path = os.path.join(
                bundle_path, service_name, PureWindowsPath(module_file).as_posix()
            )
            if not os.path.isfile(module_file_path):
                module_file_path = os.path.join(
                    bundle_path, PureWindowsPath(module_file).as_posix()
                )

    if not os.path.isfile(module_file_path):
        raise BentoMLException(
            "Can not locate module_file {} in saved bundle {}".format(
                module_file, bundle_path
            )
        )

    return module_file_path


@resolve_remote_bundle
def load_bento_service_class(bundle_path):
    """
    Load a BentoService class from saved bundle in given path

    :param bundle_path: A path to Bento files generated from BentoService#save,
        #save_to_dir, or the path to pip installed BentoService directory
    :return: BentoService class
    """
    config = load_saved_bundle_config(bundle_path)
    metadata = config["metadata"]

    # Find and load target module containing BentoService class from given path
    module_file_path = _find_module_file(
        bundle_path, metadata["service_name"], metadata["module_file"]
    )

    # Prepend bundle_path to sys.path for loading extra python dependencies
    sys.path.insert(0, bundle_path)
    sys.path.insert(0, os.path.join(bundle_path, metadata["service_name"]))
    # Include zipimport modules
    zipimport_dir = os.path.join(bundle_path, metadata["service_name"], ZIPIMPORT_DIR)
    if os.path.exists(zipimport_dir):
        for p in os.listdir(zipimport_dir):
            logger.debug('adding %s to sys.path', p)
            sys.path.insert(0, os.path.join(zipimport_dir, p))

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

    if (
        model_service_class._env
        and model_service_class._env._requirements_txt_file is not None
    ):
        # Load `requirement.txt` from bundle directory instead of the user-provided
        # file path, which may only available during the bundle save process
        model_service_class._env._requirements_txt_file = os.path.join(
            bundle_path, "requirements.txt"
        )

    return model_service_class


@resolve_remote_bundle
def safe_retrieve(bundle_path: str, target_dir: str):
    """Safely retrieve bento service to local path

    Args:
        bundle_path (:obj:`str`):
            The path that contains saved BentoService bundle, supporting
            both local file path and s3 path
        target_dir (:obj:`str`):
            Where the service contents should end up.

    Returns:
        :obj:`str`: location of safe local path
    """
    shutil.copytree(bundle_path, target_dir)


@resolve_remote_bundle
def load_from_dir(bundle_path):
    """Load bento service from local file path or s3 path

    Args:
        bundle_path (str): The path that contains saved BentoService bundle,
            supporting both local file path and s3 path

    Returns:
        bentoml.service.BentoService: a loaded BentoService instance
    """
    svc_cls = load_bento_service_class(bundle_path)
    return svc_cls()


@resolve_remote_bundle
def load_bento_service_api(bundle_path, api_name=None):
    bento_service = load_from_dir(bundle_path)
    return bento_service.get_inference_api(api_name)
