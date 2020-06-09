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

from setuptools import sandbox

from bentoml.configuration import _is_pypi_release

from bentoml.exceptions import BentoMLException
from bentoml.bundler.py_module_utils import copy_used_py_modules
from bentoml.bundler.templates import (
    BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE,
    MANIFEST_IN_TEMPLATE,
    MODEL_SERVER_DOCKERFILE_CPU,
    INIT_PY_TEMPLATE,
    BENTO_INIT_SH_TEMPLATE,
    DOCKER_ENTRYPOINT_SH,
)
from bentoml.utils.usage_stats import track_save
from bentoml.bundler.config import SavedBundleConfig


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

    for artifact in bento_service._artifacts:
        if artifact.name not in bento_service._packed_artifacts:
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
    if bento_service.artifacts:
        bento_service.artifacts.save(module_base_path)

    # write conda environment, requirement.txt
    bento_service.env.save(path, bento_service)

    # TODO: add bentoml.find_packages helper for more fine grained control over this
    # process, e.g. packages=find_packages(base, [], exclude=[], used_module_only=True)
    # copy over all custom model code
    module_name, module_file = copy_used_py_modules(
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
    with open(os.path.join(path, "Dockerfile"), "w") as f:
        f.write(
            MODEL_SERVER_DOCKERFILE_CPU.format(
                docker_base_image=bento_service._env._docker_base_image
            )
        )

    # write docker-entrypoint.sh
    docker_entrypoint_sh_file = os.path.join(path, "docker-entrypoint.sh")
    with open(docker_entrypoint_sh_file, "w") as f:
        f.write(DOCKER_ENTRYPOINT_SH)
    # chmod +x docker-entrypoint.sh
    st = os.stat(docker_entrypoint_sh_file)
    os.chmod(docker_entrypoint_sh_file, st.st_mode | stat.S_IEXEC)

    # write bento init sh for install targz bundles
    bentoml_init_script_file = os.path.join(path, 'bentoml_init.sh')
    with open(bentoml_init_script_file, 'w') as f:
        f.write(BENTO_INIT_SH_TEMPLATE)
    # chmod +x bentoml_init_script file
    st = os.stat(bentoml_init_script_file)
    os.chmod(bentoml_init_script_file, st.st_mode | stat.S_IEXEC)

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


def _bundle_local_bentoml_if_installed_from_source(bundle_path):
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
    if not _is_pypi_release() and os.path.isfile(bentoml_setup_py):
        logger.info(
            "Detect BentoML installed in development model, copying local BentoML "
            "module file to target saved bundle path"
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

        sandbox.run_setup(
            bentoml_setup_py,
            ['sdist', '--format', 'gztar', '--dist-dir', bundle_dir_name],
        )

        # copy the generated targz to saved bundle directory and remove it from
        # bentoml module directory
        dest_dir = os.path.join(bundle_path, 'bundled_pip_dependencies')
        shutil.copytree(source_dir, dest_dir)

        # clean up sdist build files
        shutil.rmtree(source_dir)
