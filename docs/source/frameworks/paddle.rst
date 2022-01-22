PaddlePaddle
------------

| PArallel Distributed Deep LEarning: Machine Learning Framework from Industrial Practice - `Source <https://www.paddlepaddle.org.cn/>`_

Users can now use PaddlePaddle with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import random
   import numpy as np

   import bentoml
   import paddle
   import paddle.nn as nn
   from paddle.static import InputSpec

   IN_FEATURES = 13
   OUT_FEATURES = 1


   def set_random_seed(seed):
      random.seed(seed)
      np.random.seed(seed)
      paddle.seed(seed)
      paddle.framework.random._manual_program_seed(seed)


   class LinearModel(nn.Layer):
      def __init__(self):
         super(LinearModel, self).__init__()
         self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)

      @paddle.jit.to_static(input_spec=[InputSpec(shape=[IN_FEATURES], dtype="float32")])
      def forward(self, x):
         return self.fc(x)


   def train_paddle_model() -> "LinearModel":
      set_random_seed(SEED)
      model = LinearModel()
      loss = nn.MSELoss()
      adam = paddle.optimizer.Adam(parameters=model.parameters())

      train_data = paddle.text.datasets.UCIHousing(mode="train")

      loader = paddle.io.DataLoader(
         train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
      )

      model.train()
      for _ in range(EPOCH_NUM):
         for _, (feature, label) in enumerate(loader()):
               out = model(feature)
               loss_fn = loss(out, label)
               loss_fn.backward()
               adam.step()
               adam.clear_grad()
      return model

   model = train_paddle_model()
   # `save` a pretrained model to BentoML modelstore:
   tag = bentoml.paddle.save("linear_model", model, input_spec=InputSpec(shape=[IN_FEATURES], dtype="float32"))

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   loaded = bentoml.paddle.load(tag)

   # `load` with custom config:
   conf = paddle.inference.Config(
       metadata.path + "/saved_model.pdmodel", info.path + "/saved_model.pdiparams"
   )
   conf.enable_memory_optim()
   conf.set_cpu_math_library_num_threads(1)
   paddle.set_device("cpu")
   loaded_with_customs: nn.Layer = bentoml.paddle.load(tag, config=conf)

   # Load a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.paddle.load_runner(tag)

   runner.run_batch(pd_dataframe.to_numpy().astype(np.float32))

We also offer :code:`import_from_paddlehub` which enables users to import model from `PaddleHub <https://www.paddlepaddle.org.cn/hub>`_ and use it with BentoML:

.. code-block:: python

   import os

   import bentoml

   test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
   tag = bentoml.paddle.import_from_paddlehub("senta_bilstm")
   runner = bentoml.paddle.load_runner(tag, infer_api_callback="sentiment_classify")
   results = runner.run_batch(None, texts=test_text, use_gpu=False, batch_size=1)

   assert results[0]["positive_probs"] == 0.9407
   assert results[1]["positive_probs"] == 0.02

   # import from local paddle hub module, refers to
   # https://paddlehub.readthedocs.io/en/release-v2.1/index.html
   senta_path = os.path.join(current_dir, "senta_test")

   # save given paddlehub module to BentoML modelstore
   tag = bentoml.paddle.import_from_paddlehub(senta_path)

   # load back into memory:
   module = bentoml.paddle.load(tag)
   assert module.sentiment_classify(texts=text)[0]["sentiment"] == "negative"

.. note::
   You can find more examples for **PaddlePaddle** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.paddle

.. autofunction:: bentoml.paddle.save

.. autofunction:: bentoml.paddle.load

.. autofunction:: bentoml.paddle.load_runner

.. autofunction:: bentoml.paddle.import_from_paddlehub