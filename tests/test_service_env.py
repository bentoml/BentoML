import os
import stat
import pytest

import psutil  # noqa # pylint: disable=unused-import

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact


def test_pip_dependencies_env():
    @bentoml.env(pip_dependencies=["numpy"])
    class ServiceWithString(bentoml.BentoService):
        @bentoml.api(input=DataframeInput())
        def predict(self, df):
            return df

    service_with_string = ServiceWithString()
    assert 'numpy' in service_with_string.env._pip_dependencies

    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput())
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    assert 'numpy' in service_with_list.env._pip_dependencies
    assert 'pandas' in service_with_list.env._pip_dependencies
    assert 'torch' in service_with_list.env._pip_dependencies


def test_service_env_pip_dependencies(tmpdir):
    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput())
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    service_with_list.save_to_dir(str(tmpdir))

    requirements_txt_path = os.path.join(str(tmpdir), 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        module_list = saved_requirements.decode('utf-8').split('\n')
        assert 'numpy' in module_list
        assert 'pandas' in module_list
        assert 'torch' in module_list


def test_artifact_pip_dependencies(tmpdir):
    @bentoml.artifacts([SklearnModelArtifact('model')])
    @bentoml.env(pip_dependencies=['scikit-learn==0.23.0'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(input=DataframeInput())
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
        @bentoml.api(input=DataframeInput())
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
        @bentoml.api(input=DataframeInput())
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
        @bentoml.api(input=DataframeInput())
        def predict(self, df):
            return df

    service_with_setup = ServiceWithSetup()
    assert 'continuumio/miniconda3:4.8.0' in service_with_setup.env._docker_base_image
