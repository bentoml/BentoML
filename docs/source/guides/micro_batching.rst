Understanding BentoML adaptive micro batching
=============================================

1. The overall architecture of BentoML's micro-batching server
==============================================================

1.1 Why micro batching matters
------------------------------

   While serving a TensorFlow model, batching individual model
   inference requests together can be important for performance. In
   particular, batching is necessary to unlock the high throughput
   promised by hardware accelerators such as GPUs.

   -- `tensorflow/serving <https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md>`__

Plus, under BentoML's architecture, the HTTP handling and data
preprocessing procedure will also benefit from micro-batching.

1.2 Architecture & Data Flow
----------------------------

.. raw:: html
   :file: ../_static/img/batching-arch.svg

1.3 Parameters & Concepts of micro batching
-------------------------------------------

-  **inbound requests**: requests from user clients
-  **outbound requests**: requests to upstream model servers
-  ``mb_max_batch_size`` The maximum size of any batch. This parameter
   governs the throughput/latency tradeoff, and also avoids having
   batches that are so large they exceed some resource constraint (e.g.
   GPU memory to hold a batch's data). Default: 1000.
-  ``mb_max_latency`` The latency goal of your service in milliseconds.
   Default: 10000.
-  **outbound semaphore:** The semaphore represents the degree of
   parallelism, i.e. the maximum number of batches processed
   concurrently. **It is set automatically** when launching the bento
   service as the same number of model server workers.
-  **Estimated time**: Estimated time for model server to execute a
   batch. Inferred from historical data and current batch size in queue.

1.4 Sequence & How it works
---------------------------

Take bento service with single API and —workers=1 as example

.. raw:: html
   :file: ../_static/img/batching-seq.svg


To achieve optimal efficiency, the CORK dispatcher performs a adaptive
control to **cork**/**release** inbound requests. The releasing happens
when:

-  meets one of the following conditions:

   -  the **waited time** + **estimated time** exceeds
      ``mb_max_latency`` OR
   -  it is not worth to wait next inbound request \*

-  AND the outbound semaphore is not locked

A large ``mb_max_latency`` didn't represents that each request will be
responded in this latency. The algorithm will determine a adaptive wait
time between 0 and the ``mb_max_latency``. But when under excessive
request pressure, more response time will reach the ``mb_max_latency``.

In each releasing, the count of released requests is decided by
algorithm, but less than ``mb_max_batch_size``.

If the outbound semaphore is still locked, requests may be canceled once
reached ``mb_max_latency``.

1.5 The main design decisions and tradeoffs
-------------------------------------------

Throughput and latency are most concerned for API servers. BentoML will
fine-tune batches **automatically** to(in the order priority):

-  Ensure the user defined constraint of ``mb_max_batch_size`` and
   ``mb_max_latency``.
-  Maximum the Throughput
-  Minimum the average Latency

2. parameter tuning best practices & recommendations
====================================================

Different from TensorFlow Serving, BentoML will **automatically adjust**
the batch size and wait timeout, balancing the maximum throughput and
latency. It will respond to the fluctuations of server loading.

.. code:: python

    class MovieReviewService(bentoml.BentoService):
        @bentoml.api(input=DataframeInput(),
                     mb_max_latency=10000, mb_max_batch_size=1000)
        def predict(self, inputs):
                    pass

``mb_max_batch_size`` is 1000 by default and ``mb_max_latency`` is 10000
by default.

-  If the RAM of GPU only allowed input with 100 batch size, then you
   could set ``mb_max_batch_size`` to 100
-  If the clients using your API has the request timeout 200ms, then you
   could set ``mb_max_latency`` to 200.
-  If you know the executing of your model is very slow (for example,
   the latency is more than 100ms), then enlarging the
   ``mb_max_latency`` to 10 \* 100ms will help to achieve higher
   throughput.

3. How to implement batch mode for custom input adapters
==================================================

TL;DR: Implement the method ``handle_batch_request(requests)``
following existent input adapters.

The batching service is HTTP request-wise now, which is mostly
transparent for developers. The only difference between
``handle_batch_request`` and ``handle_request`` is:

-  the input parameter is a list of request object
-  the return value should be a list of response object

To maximize the benefit of micro-batching, remember to use the batch
alternative of each operation from the beginning. For example, each
``pd.read_csv/read_json`` take constantly 2ms, so code like this

.. code:: python

    def handle_batch_request(self, requests):
        dfs = []
        for req in requests:
            dfs.append(pd.read_csv(req.body))
        # ...

will be O(N) in time complexity. Thus we implemented an nearly O(1)
function to concat DataFrame CSV strings, so that all DataFrames in
requests could be loaded by calling ``pd.read_csv`` once.

4. Comparison
=============

4.1 TensorFlow Serving
----------------------

Tensorflow Serving employed similar approach to batch individual
requests together. But the parameters of batching scheduling is static.
Assume your model had 1 ms latency. If you enabled batching and
configure it with ``batch_timeout_micros = 300 * 1000``, whether
necessary or not, the latency of every request now would be 300ms + 1ms.

You will need to fine-tune these parameters by experiments before
deployment. Once deployed, it won't change anymore.

   The best values to use for the batch scheduling parameters depend on
   your model, system and environment, as well as your throughput and
   latency goals. Choosing good values is best done via experiments.
   Here are some guidelines that may be helpful in selecting values to
   experiment with.

   -- `tensorflow/serving <https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md#performance-tuning>`__

4.2 Clipper
-----------

Clipper applied a combination of TCP Nagle and AIMD algorithm. This
approach is more similar with BentoML, the difference is scheduling
algorithm and the goal of optimization.

   To automatically find the optimal maximum batch size for each model
   container we employ an additive-increase-multiplicative-decrease
   (AIMD) scheme.

   -- `Clipper: A Low-Latency Online Prediction Serving System <https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf>`__

Clipper has parameter SLO(similar with mb\_max\_latency), the
optimization goal of AIMD is to maximize the throughput under the bound
of SLO.

Therefore, for most cases, Clipper have higher latency than BentoML,
which also means it's able to serve less users at same time.

