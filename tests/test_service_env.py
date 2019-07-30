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

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(requirements_txt="""numpy
pandas
torch""")
    class ServiceWithString(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    @bentoml.artifacts([bentoml.artifact.PickleArtifact('model')])
    @bentoml.env(requirements_txt=['numpy', 'pandas', 'torch'])
    class ServiceWithList(bentoml.BentoService):
        @bentoml.api(bentoml.handlers.DataframeHandler)
        def predict(self, df):
            return df

    service_with_file = ServiceWithFile.pack(model=model)
    assert service_with_file.env._requirements_txt.decode('utf-8') == final_string

    service_with_string = ServiceWithString.pack(model=model)
    assert service_with_string.env._requirements_txt.decode('utf-8') == final_string

    service_with_list = ServiceWithList.pack(model=model)
    assert service_with_list.env._requirements_txt.decode('utf-8') == final_string
