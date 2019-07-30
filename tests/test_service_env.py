import os

import bentoml


def test_requirement_txt_env(tmpdir):
    req_txt_file = tmpdir.join("requirements.txt")
    model = ''
    final_string = "numpy\npandas\ntorch"
    with open(str(req_txt_file), 'wb') as f:
        f.write(b"numpy\npandas\ntorch")

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(requirements_txt=str(req_txt_file))
    class ServiceWithFile(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    service_with_file = ServiceWithFile.pack(model=model)
    assert service_with_file.env._requirements_txt.decode('utf-8') == final_string


def test_pip_dependencies_env(tmpdir):
    model = ''

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(pip_dependencies="numpy")
    class ServiceWithString(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    service_with_string = ServiceWithString.pack(model=model)
    assert 'numpy' in service_with_string.env._pip_dependencies

    service_with_list = ServiceWithList.pack(model=model)
    assert len(service_with_list.env._pip_dependencies) == 3


def test_pip_dependencies_with_archive(tmpdir):
    model = ''

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(pip_dependencies=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    service_with_list = ServiceWithList.pack(model=model)
    saved_path = service_with_list.save(tmpdir)

    requirements_txt_path = os.path.join(saved_path, 'requirements.txt')
    with open(requirements_txt_path, 'rb') as f:
        saved_requirements = f.read()
        print(saved_requirements)
        module_list = saved_requirements.decode('utf-8').split('\n')
        print(module_list)
        assert len(module_list) == 3
