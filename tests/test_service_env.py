import os

import bentoml
from bentoml.handlers import DataframeHandler


def test_requirement_txt_env(tmpdir):
    req_txt_file = tmpdir.join("requirements.txt")
    with open(str(req_txt_file), 'wb') as f:
        f.write(b"numpy\npandas\ntorch")

    @bentoml.env(requirements_txt=str(req_txt_file))
    class ServiceWithFile(bentoml.BentoService):
        @bentoml.api(DataframeHandler)
        def predict(self, df):
            return df

    service_with_file = ServiceWithFile()
    assert 'numpy' in service_with_file.env._pip_dependencies
    assert 'pandas' in service_with_file.env._pip_dependencies
    assert 'torch' in service_with_file.env._pip_dependencies

    saved_path = service_with_file.save('/tmp')
    with open(os.path.join(saved_path, 'requirements.txt'), 'rb') as f:
        content = f.read().decode('utf-8')
        assert 'numpy' in content
        assert 'pandas' in content
        assert 'torch' in content


def test_pip_dependencies_env():
    @bentoml.env(pip_dependencies="numpy")
    class ServiceWithString(bentoml.BentoService):
        @bentoml.api(DataframeHandler)
        def predict(self, df):
            return df

    service_with_string = ServiceWithString()
    assert 'numpy' in service_with_string.env._pip_dependencies

    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(DataframeHandler)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    assert 'numpy' in service_with_list.env._pip_dependencies
    assert 'pandas' in service_with_list.env._pip_dependencies
    assert 'torch' in service_with_list.env._pip_dependencies


def test_service_env_pip_dependencies(tmpdir):
    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(DataframeHandler)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList()
    saved_path = service_with_list.save(str(tmpdir))

    requirements_txt_path = os.path.join(saved_path, 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        module_list = saved_requirements.decode('utf-8').split('\n')
        assert 'numpy' in module_list
        assert 'pandas' in module_list
        assert 'torch' in module_list
