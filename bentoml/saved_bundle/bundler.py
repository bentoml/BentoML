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
import importlib
import os
import shutil
import stat
import logging


from bentoml.configuration import _is_pip_installed_bentoml

from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle.local_py_modules import copy_local_py_modules
from bentoml.saved_bundle.templates import (
    BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE,
    MANIFEST_IN_TEMPLATE,
    MODEL_SERVER_DOCKERFILE_CPU,
    INIT_PY_TEMPLATE,
)
from bentoml.utils.usage_stats import track_save
from bentoml.saved_bundle.config import SavedBundleConfig


DEFAULT_SAVED_BUNDLE_README = """\
# Generated BentoService bundle - {}:{}

This is a ML Service bundle created with BentoML, it is not recommended to edit
code or files contained in this directory. Instead, edit the code that uses BentoML
to create this bundle, and save a new BentoService bundle.
"""

logger = logging.getLogger(__name__)


def save_to_dir(bento_service, path, version=None, silent=False):
    """Save given BentoService along with all its artifacts, source code and
    dependencies to target file path, assuming path exist and empty. If target path
    is not empty, this call may override existing files in the given path.

    :param bento_service (bentoml.service.BentoService): a Bento Service instance
    :param path (str): Destination of where the bento service will be saved
    :param version (str): Override the service version with given version string
    :param silent (boolean): whether to hide the log message showing target save path
    """
    track_save(bento_service)

    from bentoml.service import BentoService

    if not isinstance(bento_service, BentoService):
        raise BentoMLException(
            "save_to_dir only work with instance of custom BentoService class"
        )

    if version is not None:
        # If parameter version provided, set bento_service version
        # Otherwise it will bet set the first time the `version` property get accessed
        bento_service.set_version(version)

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
    bento_service.env.save(path, bento_service)

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

    if not silent:
        logger.info(
            "BentoService bundle '%s:%s' created at: %s",
            bento_service.name,
            bento_service.version,
            path,
        )


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
