Adding Custom Model Artifact
============================

BentoML integrates with the most popular machine learning frameworks. For the ML framework yet to integrate with BentoML,
BentoML provides model artifact customizing...

The guide will demonstrate how to create a custom model artifact class, and then use it in BentoService for prediction

----------------------
Create custom Artifact
----------------------

The following code creates a subclass from the `BentoServiceArtifact`. It implements how to
save and load the model.  In the `pack` method, the model class does validation to make sure
the model is valid.  It uses `cloudpickle` to `save` and `load`.


.. code-block:: python

    # my_model_artifact.py

    import os
    import json
    from bentoml.utils import cloudpickle
    from bentoml.exceptions import InvalidArgument
    from bentoml.service.artifacts import BentoServiceArtifact

    class MyModelArtifact(BentoServiceArtifact):
        def __init__(self, name):
            super(MyModelArtifact, self).__init__(name)
            self._model = None

        def pack(self, model, metadata=None):
            if isinstance(model, dict) is not True:
                raise InvalidArgument('MyModelArtifact only support dict')
            if model.get('foo', None) is None:
                raise KeyError('"foo" is not available in the model')
            self._model = model
            return self

        def get(self):
            return self._model

        def save(self):
            with open(self._file_path, 'wb') as file:
                cloudpickle.dump(self._model, file)

        def load(self, path):
            with open(self._file_path(path), 'rb') as file:
                model = cloudpickle.load(file)
            return self.pack(model)

        def _file_path(self, base_path):
            return os.path.join(base_path, self.name + '.json')


-----------------------------------------------------
Define and save BentoService with the custom Artifact
-----------------------------------------------------

.. code-block:: python

    # my_bento_service.py

    from my_model_artifact import MyModelArtifact
    from bentoml import BentoService, env, api, artifacts
    from bentoml.adapters import JsonInputAdapter
    import bentoml

    @env(infer_pip_packages=True)
    @artifacts([MyModelArtifact('test_model')])
    class MyService(bentoml.BentoService):

        @api(input=JsonInput, batch=False)
        def predict(self, input_data):
            result = input_data['foo'] + self.artifacts.test_model['bar']
            return {'result': result}


.. code-block:: python

    from my_bento_service import MyService

    svc = MyService()
    model = {'foo': 2}
    svc.pack('test_model', model)
    svc.save()

----------------------
Test with example data
----------------------

.. code-block:: shell

    $ bentoml serve MyService:latest


In another terminal to make a `curl` request

.. code-block:: shell

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"bar": 1}' \
      http://localhost:5000/predict

    # Output
    HTTP/1.0 400 BAD REQUEST
    X-Request-Id: cb63a61e-dc2a-4e12-a91c-8b15316a99df
    Content-Type: text/html; charset=utf-8
    Content-Length: 20
    Server: Werkzeug/0.15.4 Python/3.7.3
    Date: Tue, 16 Mar 2021 01:47:38 GMT

    '{"result": 3}'%

