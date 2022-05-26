=============
Using Runners
=============

What is Runner?
---------------

Runner instances also provides a :code:`async_run` method, which can be used in
:ref:`async endpoints <concepts/service:Sync vs Async APIs>`:



Pre-built Model Runners
-----------------------

.. code:: python

  @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
  async def classify(input_series: np.ndarray) -> np.ndarray:
     result = await iris_clf_runner.predict.async_run(input_series)
     return result


Custom Runner
-------------

nltk example

class MyRunnable(bentoml.Runnable):

    SUPPORT_GPU = True
    SUPPORT_MULTI_THREADING = True

	def __init__(self, foo, bar):
			...

	@bentoml.Runnable.method(
	  batchable=True,
	  batch_dim=0,
		input_spec: ... # optional
		output_spec: ... # optional
	)
	def predict(self, input_data):
			...

my_runner = bentoml.Runner(MyRunnable, runnable_init_params={})


Testing Custom Runner
^^^^^^^^^^^^^^^^^^^^^

Testing Runner script:

my_runner = benotml.pytorch.get(tag).to_runner(..)

os.environ["CUDA_VISIBLE_DEVICES"] = None  # if needed
my_runner.init_local()
# warning user: for testing purpose only
# warning user: resource configs are not respected

my_runner.predict.run( test_input_df )


Custom Model Runner
^^^^^^^^^^^^^^^^^^^


bento_model = bentoml.models.get()

class MyRunnable(bentoml.Runnable):

    SUPPORT_GPU = True
    SUPPORT_MULTI_THREADING = True

	def __init__(self, foo, bar):
			...

	@bentoml.Runnable.method(
	  batchable=True,
	  batch_dim=0,
		input_spec: ... # optional
		output_spec: ... # optional
	)
	def predict(self, input_data):
			...

my_runner = bentoml.Runner(MyRunnable, runnable_init_params={}, models=[bento_model])


Runner Options
--------------

my_runner = bentoml.Runner(
	MyRunnable,
	init_params={"foo": foo, "bar": bar},
	name="custom_runner_name",
	strategy=None, # default strategy will be selected depending on the SUPPORT_GPU and SUPPORT_CPU_MULTI_THREADING flag on runnable
	models=[..],

	# below are also configurable via config file:

	# default configs:
	cpu=4,
	nvidia_gpu=1
	custom_resources={..} # reserved API for supporting custom accelerators, a custom scheduling strategy will be needed to support new hardware types
	max_batch_size=..  # default max batch size will be applied to all run methods, unless override in the runnable_method_configs
	max_latency_ms=.. # default max latency will be applied to all run methods, unless override in the runnable_method_configs

	runnable_method_configs=[
		{
			method_name="predict",
			max_batch_size=..,
			max_latency_ms=..,
		}
	],
)



Runner(
            self.to_runnable(),
            name=name if name != "" else self.tag.name,
            models=[self],
            cpu=cpu,
            nvidia_gpu=nvidia_gpu,
            custom_resources=custom_resources,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=method_configs,
        )


Runner Resource Config
----------------------

runners:
  - name: iris_clf
    cpu: 4
    nvidia_gpu: 0  # requesting 0 GPU
    max_batch_size: 20
  - name: my_custom_runner
	cpu: 2
	nvidia_gpu: 2  # requesting 2 GPUs
	runnable_method_configs:
      - name: "predict"
		max_batch_size: 10
		max_latency_ms: 500



Runner and BentoServer Architecture
-----------------------------------


Distributed Runner with Yatai
-----------------------------
