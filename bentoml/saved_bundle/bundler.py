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

import datetime
import glob
import gzip
import importlib
import io
import json
import logging
import os
import shutil
import stat
import tarfile
from urllib.parse import urlparse

from typing import TYPE_CHECKING
import requests

from bentoml.configuration import _is_pip_installed_bentoml
from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle.local_py_modules import (
    copy_local_py_modules,
    copy_zip_import_archives,
)
from bentoml.saved_bundle.loader import _is_remote_path
from bentoml.saved_bundle.templates import (
    BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE,
    INIT_PY_TEMPLATE,
    MANIFEST_IN_TEMPLATE,
    MODEL_SERVER_DOCKERFILE_CPU,
)
from bentoml.utils import is_gcs_url, is_s3_url, archive_directory_to_tar
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.usage_stats import track_save
from bentoml.saved_bundle.config import SavedBundleConfig
from bentoml.saved_bundle.pip_pkg import get_zipmodules, ZIPIMPORT_DIR
from bentoml.utils.open_api import get_open_api_spec_json

if TYPE_CHECKING:
    from bentoml.service import BentoService

DEFAULT_SAVED_BUNDLE_README = """\
# Generated BentoService bundle - {}:{}

This is a ML Service bundle created with BentoML, it is not recommended to edit
code or files contained in this directory. Instead, edit the code that uses BentoML
to create this bundle, and save a new BentoService bundle.
"""

logger = logging.getLogger(__name__)


def _write_bento_content_to_dir(bento_service: "BentoService", path: str):
    if not os.path.exists(path):
        raise BentoMLException("Directory '{}' not found".format(path))

    for artifact in bento_service.artifacts.get_artifact_list():
        if not artifact.packed:
            logger.warning(
                "Missing declared artifact '%s' for BentoService '%s'",
                artifact.name,
                bento_service.name,
            )
    module_base_path = os.path.join(path, bento_service.name)
    try:
        os.mkdir(module_base_path)
    except FileExistsError:
        raise BentoMLException(
            f"Existing module file found for BentoService {bento_service.name}"
        )

    # write README.md with custom BentoService's docstring if presented
    saved_bundle_readme = DEFAULT_SAVED_BUNDLE_README.format(
        bento_service.name, bento_service.version
    )
    if bento_service.__class__.__doc__:
        saved_bundle_readme += "\n"
        saved_bundle_readme += bento_service.__class__.__doc__.strip()

    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(saved_bundle_readme)

    # save all model artifacts to 'base_path/name/artifacts/' directory
    bento_service.artifacts.save(module_base_path)

    # write conda environment, requirement.txt
    bento_service.env.infer_pip_packages(bento_service)
    bento_service.env.save(path)

    # Copy all local python modules used by the module containing the `bento_service`'s
    # class definition to saved bundle directory
    module_name, module_file = copy_local_py_modules(
        bento_service.__class__.__module__, os.path.join(path, bento_service.name)
    )

    # create __init__.py
    with open(os.path.join(path, bento_service.name, "__init__.py"), "w") as f:
        f.write(
            INIT_PY_TEMPLATE.format(
                service_name=bento_service.name,
                module_name=module_name,
                pypi_package_version=bento_service.version,
            )
        )

    # write setup.py, this make saved BentoService bundle pip installable
    setup_py_content = BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE.format(
        name=bento_service.name,
        pypi_package_version=bento_service.version,
        long_description=saved_bundle_readme,
    )
    with open(os.path.join(path, "setup.py"), "w") as f:
        f.write(setup_py_content)

    with open(os.path.join(path, "MANIFEST.in"), "w") as f:
        f.write(MANIFEST_IN_TEMPLATE.format(service_name=bento_service.name))

    # write Dockerfile
    logger.debug("Using Docker Base Image %s", bento_service._env._docker_base_image)
    with open(os.path.join(path, "Dockerfile"), "w") as f:
        f.write(
            MODEL_SERVER_DOCKERFILE_CPU.format(
                docker_base_image=bento_service._env._docker_base_image
            )
        )

    # copy custom web_static_content if enabled
    if bento_service.web_static_content:
        src_web_static_content_dir = os.path.join(
            os.getcwd(), bento_service.web_static_content
        )
        if not os.path.isdir(src_web_static_content_dir):
            raise BentoMLException(
                f'web_static_content directory {src_web_static_content_dir} not found'
            )
        dest_web_static_content_dir = os.path.join(
            module_base_path, 'web_static_content'
        )
        shutil.copytree(src_web_static_content_dir, dest_web_static_content_dir)

    # Copy docker-entrypoint.sh
    docker_entrypoint_sh_file_src = os.path.join(
        os.path.dirname(__file__), "docker-entrypoint.sh"
    )
    docker_entrypoint_sh_file_dst = os.path.join(path, "docker-entrypoint.sh")
    shutil.copyfile(docker_entrypoint_sh_file_src, docker_entrypoint_sh_file_dst)
    # chmod +x docker-entrypoint.sh
    st = os.stat(docker_entrypoint_sh_file_dst)
    os.chmod(docker_entrypoint_sh_file_dst, st.st_mode | stat.S_IEXEC)

    # copy bentoml-init.sh for install targz bundles
    bentoml_init_sh_file_src = os.path.join(
        os.path.dirname(__file__), "bentoml-init.sh"
    )
    bentoml_init_sh_file_dst = os.path.join(path, "bentoml-init.sh")
    shutil.copyfile(bentoml_init_sh_file_src, bentoml_init_sh_file_dst)
    # chmod +x bentoml_init_script file
    st = os.stat(bentoml_init_sh_file_dst)
    os.chmod(bentoml_init_sh_file_dst, st.st_mode | stat.S_IEXEC)

    # write bentoml.yml
    config = SavedBundleConfig(bento_service)
    config["metadata"].update({"module_name": module_name, "module_file": module_file})

    config.write_to_path(path)
    # Also write bentoml.yml to module base path to make it accessible
    # as package data after pip installed as a python package
    config.write_to_path(module_base_path)

    bundled_pip_dependencies_path = os.path.join(path, 'bundled_pip_dependencies')
    _bundle_local_bentoml_if_installed_from_source(bundled_pip_dependencies_path)
    # delete mtime and sort file in tarballs to normalize the checksums
    for tarball_file_path in glob.glob(
        os.path.join(bundled_pip_dependencies_path, '*.tar.gz')
    ):
        normalize_gztarball(tarball_file_path)

    # write open-api-spec json file
    with open(os.path.join(path, "docs.json"), "w") as f:
        json.dump(get_open_api_spec_json(bento_service), f, indent=2)


def save_to_dir(
    bento_service: "BentoService", path: str, version: str = None, silent: bool = False
) -> None:
    """
    Save given :class:`~bentoml.BentoService` along with all its artifacts,
    source code and dependencies to target file path, assuming path
    exist and empty. If target path is not empty, this call may override
    existing files in the given path.

    Args:
        bento_service (:class:`~bentoml.service.BentoService`):
            a BentoService instance
        path (`str`):
            Destination of where the bento service will be saved. The
            destination can be local path or remote path. The remote
            path supports both AWS S3('s3://bucket/path') and
            Google Cloud Storage('gs://bucket/path').
        version (`str`, `optional`):
            Override the service version with given version string
        silent (`bool`, `optional`):
            whether to hide the log message showing target save path
    """
    track_save(bento_service)

    from bentoml.service import BentoService

    if not isinstance(bento_service, BentoService):
        raise BentoMLException(
            "save_to_dir only works with instances of custom BentoService class"
        )

    if version is not None:
        # If parameter version provided, set bento_service version
        # Otherwise it will bet set the first time the `version` property get accessed
        bento_service.set_version(version)

    if _is_remote_path(path):
        # If user provided path is an remote location, the bundle will first save to
        # a temporary directory and then upload to the remote location
        logger.info(
            'Saving bento to an remote path. BentoML will first save the bento '
            'to a local temporary directory and then upload to the remote path.'
        )
        with TempDirectory() as temp_dir:
            _write_bento_content_to_dir(bento_service, temp_dir)
            with TempDirectory() as tarfile_dir:
                tarfile_path, tarfile_name = archive_directory_to_tar(
                    temp_dir, tarfile_dir, bento_service.name
                )
                _upload_file_to_remote_path(path, tarfile_path, tarfile_name)
    else:
        _write_bento_content_to_dir(bento_service, path)

    copy_zip_import_archives(
        os.path.join(path, bento_service.name, ZIPIMPORT_DIR),
        bento_service.__class__.__module__,
        list(get_zipmodules().keys()),
        bento_service.env._zipimport_archives or [],
    )

    if not silent:
        logger.info(
            "BentoService bundle '%s:%s' created at: %s",
            bento_service.name,
            bento_service.version,
            path,
        )


def normalize_gztarball(file_path: str):
    MTIME = datetime.datetime(2000, 1, 1).timestamp()
    tar_io = io.BytesIO()

    with tarfile.open(file_path, "r:gz") as f:
        with tarfile.TarFile("bundle.tar", mode='w', fileobj=tar_io) as nf:
            names = sorted(f.getnames())
            for name in names:
                info = f.getmember(name)
                info.mtime = MTIME
                nf.addfile(info, f.extractfile(name))

    with open(file_path, "wb") as nf:
        with gzip.GzipFile("bundle.tar.gz", mode="w", fileobj=nf, mtime=MTIME) as gf:
            gf.write(tar_io.getvalue())


def _bundle_local_bentoml_if_installed_from_source(target_path):
    """
    if bentoml is installed in editor mode(pip install -e), this will build a source
    distribution with the local bentoml fork and add it to saved BentoService bundle
    path under bundled_pip_dependencies directory
    """

    # Find bentoml module path
    (module_location,) = importlib.util.find_spec('bentoml').submodule_search_locations

    bentoml_setup_py = os.path.abspath(os.path.join(module_location, '..', 'setup.py'))

    # this is for BentoML developer to create BentoService containing custom develop
    # branches of BentoML library, it is True only when BentoML module is installed in
    # development mode via "pip install --editable ."
    if not _is_pip_installed_bentoml() and os.path.isfile(bentoml_setup_py):
        logger.info(
            "Detected non-PyPI-released BentoML installed, copying local BentoML module"
            "files to target saved bundle path.."
        )

        # Create tmp directory inside bentoml module for storing the bundled
        # targz file. Since dist-dir can only be inside of the module directory
        bundle_dir_name = '__bentoml_tmp_sdist_build'
        source_dir = os.path.abspath(
            os.path.join(module_location, '..', bundle_dir_name)
        )

        if os.path.isdir(source_dir):
            shutil.rmtree(source_dir, ignore_errors=True)
        os.mkdir(source_dir)

        from setuptools import sandbox

        sandbox.run_setup(
            bentoml_setup_py,
            ['-q', 'sdist', '--format', 'gztar', '--dist-dir', bundle_dir_name],
        )

        # copy the generated targz to saved bundle directory and remove it from
        # bentoml module directory
        shutil.copytree(source_dir, target_path)

        # clean up sdist build files
        shutil.rmtree(source_dir)


def _upload_file_to_remote_path(remote_path, file_path: str, file_name: str):
    """Upload file to remote path
    """
    parsed_url = urlparse(remote_path)
    bucket_name = parsed_url.netloc
    object_prefix_path = parsed_url.path.lstrip('/')
    object_path = f'{object_prefix_path}/{file_name}'
    if is_s3_url(remote_path):
        try:
            import boto3
        except ImportError:
            raise BentoMLException(
                '"boto3" package is required for saving bento to AWS S3 bucket'
            )
        s3_client = boto3.client('s3')
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(f, bucket_name, object_path)
    elif is_gcs_url(remote_path):
        try:
            from google.cloud import storage
        except ImportError:
            raise BentoMLException(
                '"google.cloud" package is required for saving bento to Google '
                'Cloud Storage'
            )
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.upload_from_filename(file_path)
    else:
        http_response = requests.put(remote_path)
        if http_response.status_code != 200:
            raise BentoMLException(
                f'Error uploading BentoService to {remote_path} '
                f'{http_response.status_code}'
            )
