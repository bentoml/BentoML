import os
import stat

import psutil  # noqa # pylint: disable=unused-import
import pytest

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact


def test_pip_packages_env_with_legacy_api():
    @bentoml.env(pip_packages=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    assert 'numpy' in service_with_list.env._pip_packages
    assert 'pandas' in service_with_list.env._pip_packages
    assert 'torch' in service_with_list.env._pip_packages


def test_pip_packages_env():
    @bentoml.env(pip_packages=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    assert 'numpy' in service_with_list.env._pip_packages
    assert 'pandas' in service_with_list.env._pip_packages
    assert 'torch' in service_with_list.env._pip_packages


def test_service_env_pip_packages(tmpdir):
    @bentoml.env(pip_packages=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    service_with_list.save_to_dir(str(tmpdir))

    requirements_txt_path = os.path.join(str(tmpdir), 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        content = saved_requirements.decode('utf-8')
        assert 'numpy' in content
        assert 'pandas' in content
        assert 'torch' in content


def test_service_env_pip_install_options(tmpdir):
    sample_index_url = "https://pip.my_pypi_index.com"
    sample_trusted_host = "https://pip.my_pypi_index.com"
    sample_extra_index_url = "https://pip.my_pypi_index_ii.com"

    @bentoml.env(
        pip_packages=['numpy', 'pandas', 'torch'],
        pip_index_url=sample_index_url,
        pip_trusted_host=sample_trusted_host,
        pip_extra_index_url=sample_extra_index_url,
    )
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    service_with_list.save_to_dir(str(tmpdir))

    requirements_txt_path = os.path.join(str(tmpdir), 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        req_file = saved_requirements.decode('utf-8')
        assert 'numpy' in req_file
        assert 'pandas' in req_file
        assert 'torch' in req_file
        req_file_lines = req_file.split('\n')
        assert f'--index-url={sample_index_url}' in req_file_lines
        assert f'--trusted-host={sample_trusted_host}' in req_file_lines
        assert f'--extra-index-url={sample_extra_index_url}' in req_file_lines


def test_artifact_pip_packages(tmpdir):
    @bentoml.artifacts([SklearnModelArtifact('model')])
    @bentoml.env(pip_packages=['scikit-learn==0.23.0'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    service_with_list.save_to_dir(str(tmpdir))

    requirements_txt_path = os.path.join(str(tmpdir), 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        module_list = saved_requirements.decode('utf-8').split('\n')
        assert 'scikit-learn==0.23.0' in module_list


@pytest.mark.skipif('not psutil.POSIX')
def test_can_instantiate_setup_sh_from_file(tmpdir):
    script_path = os.path.join(tmpdir, 'script.sh')
    with open(script_path, 'w') as f:
        f.write('ls')

    @bentoml.env(setup_sh=script_path)
    class ServiceWithSetup(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_setup = ServiceWithSetup()
    service_with_setup.save_to_dir(str(tmpdir))

    setup_sh_path = os.path.join(str(tmpdir), 'setup.sh')
    assert os.path.isfile(setup_sh_path)

    st = os.stat(setup_sh_path)
    assert st.st_mode & stat.S_IEXEC

    with open(setup_sh_path, 'r') as f:
        assert f.read() == 'ls'


@pytest.mark.skipif('not psutil.POSIX')
def test_can_instantiate_setup_sh_from_txt(tmpdir):
    @bentoml.env(setup_sh='ls')
    class ServiceWithSetup(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_setup = ServiceWithSetup()
    service_with_setup.save_to_dir(str(tmpdir))

    setup_sh_path = os.path.join(str(tmpdir), 'setup.sh')
    assert os.path.isfile(setup_sh_path)

    st = os.stat(setup_sh_path)
    assert st.st_mode & stat.S_IEXEC

    with open(setup_sh_path, 'r') as f:
        assert f.read() == 'ls'


def test_docker_base_image_env():
    @bentoml.env(docker_base_image='continuumio/miniconda3:4.8.0')
    class ServiceWithSetup(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_setup = ServiceWithSetup()
    assert 'continuumio/miniconda3:4.8.0' in service_with_setup.env._docker_base_image


def test_conda_channels_n_dependencies(tmpdir):
    @bentoml.env(
        conda_channels=["bentoml-test-channel"], conda_dependencies=["bentoml-test-lib"]
    )
    class ServiceWithCondaDeps(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_string = ServiceWithCondaDeps()
    service_with_string.save_to_dir(str(tmpdir))

    from pathlib import Path
    from bentoml.utils.ruamel_yaml import YAML

    yaml = YAML()
    env_yml = yaml.load(Path(os.path.join(tmpdir, 'environment.yml')))
    assert 'conda-forge' in env_yml['channels']
    assert 'defaults' in env_yml['channels']
    assert 'bentoml-test-channel' in env_yml['channels']

    assert 'pip' in env_yml['dependencies']
    assert 'bentoml-test-lib' in env_yml['dependencies']


def test_conda_overwrite_channels(tmpdir):
    @bentoml.env(
        conda_channels=["bentoml-test-channel"],
        conda_dependencies=["bentoml-test-lib"],
        conda_overwrite_channels=True,
    )
    class ServiceWithCondaDeps(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_string = ServiceWithCondaDeps()
    service_with_string.save_to_dir(str(tmpdir))

    from pathlib import Path
    from bentoml.utils.ruamel_yaml import YAML

    yaml = YAML()
    env_yml = yaml.load(Path(os.path.join(tmpdir, 'environment.yml')))
    assert 'bentoml-test-channel' in env_yml['channels']
    assert len(env_yml['channels']) == 1


def test_conda_env_yml_file_option(tmpdir):
    conda_env_yml_file = os.path.join(tmpdir, 'environment.yml')
    with open(conda_env_yml_file, 'wb') as f:
        f.write(
            """
name: bentoml-test-conda-env
channels:
  - test-ch-1
  - test-ch-2
dependencies:
  - test-dep-1
""".encode()
        )

    @bentoml.env(
        conda_env_yml_file=conda_env_yml_file,
        conda_channels=["bentoml-test-channel"],
        conda_dependencies=["bentoml-test-lib"],
    )
    class ServiceWithCondaDeps(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return df

    service_with_string = ServiceWithCondaDeps()
    service_with_string.save_to_dir(str(tmpdir))

    from pathlib import Path
    from bentoml.utils.ruamel_yaml import YAML

    yaml = YAML()
    env_yml = yaml.load(Path(os.path.join(tmpdir, 'environment.yml')))
    assert 'test-ch-1' in env_yml['channels']
    assert 'test-ch-2' in env_yml['channels']
    assert 'bentoml-test-channel' in env_yml['channels']

    assert 'test-dep-1' in env_yml['dependencies']
    assert 'bentoml-test-lib' in env_yml['dependencies']

    assert os.path.isfile(
        Path(os.path.join(tmpdir, service_with_string.name, 'environment.yml'))
    )
