=====
Spark
=====

`Apache Spark <https://spark.apache.org/>`_ is a general-purpose distributed processing system used
for big data workloads. It allows for processing large datasets through an in-memory computation
model, which can improve the performance of big data processing tasks. It also provides a wide range
of APIs and a feature-rich set of tools for structured data processing, machine learning, and stream
processing for big-data applications.

BentoML now supports running your Bentos with batch data via Spark. The following tutorial assumes
basic understanding of BentoML. If you'd like to learn more about BentoML, see the
:ref:`BentoML tutorial <tutorial:Creating a Service>`.

Prerequisites
#############

Make sure to have at least BentoML 1.0.13 and Spark version 3.3.0 available in your system.

.. code-block:: bash

    $ pip install -U "bentoml>=1.0.13"


In addition, both BentoML and your service's dependencies (including model dependencies) must also
be installed in the Spark cluster. Most likely, the service you are hosting Spark on has its own
mechanisms for doing this. If you are using a standalone cluster, you should install those
dependencies on every node you expect to use.

Finally, we use the quickstart bento from the :doc:`aforementioned tutorial </tutorial>`. If you have
already followed that tutorial, you should already have that bento. If you have note, simply run the
following:

.. code-block:: python

    import urllib.request
    urllib.request.urlretrieve("https://bentoml-public.s3.us-west-1.amazonaws.com/quickstart/iris_classifier.bento", "iris_classifier.bento")
    bentoml.import_bento("iris_classifier.bento")

Run Bentos in Spark
###################

.. note::

    All of the following commands/APIs should work for bentos with
    :ref:`IO Descriptor <reference/api_io_descriptors:API IO Descriptors>` that support batch
    inference. Currently, those descriptors are
    :ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`,
    :ref:`bentoml.io.PandasDataFrame, and bentoml.io.PandasSeries <reference/api_io_descriptors:Tabular Data with Pandas>`.

:bdg-warning:`IMPORTANT:` your Bento API must be capable of accepting multiple inputs. For example,
``batch_classify(np.array([[input_1], [input_2]]))`` must work, and return
``np.array([[output_1], [output_2]])``. The quickstart bento supports this pattern because the iris
classifier model it contains does.

Create a PySpark SparkSession object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will be used to create a DataFrame from the input
data, and to run the batch inference job. If you're running in a notebook with spark already
(e.g. a VertexAI PySpark notebook or a Databricks Notebook), you can skip this step.

.. code-block:: python

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

Load the input data into a PySpark DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using multipart input, or your dataframe
requires column names, you must also provide a schema for your DataFrame as you load it. You can
do this using the ``spark.read.csv()`` method, which takes a file path as input and returns a
DataFrame containing the data from the file.

.. code-block:: python

    from pyspark.sql.types import StructType, StructField, FloatType, StringType
    import urllib.request

    urllib.request.urlretrieve("https://docs.bentoml.org/en/latest/_static/examples/batch/input.csv", "input.csv")

    schema = StructType([
        StructField("sepal_length", FloatType(), False),
        StructField("sepal_width", FloatType(), False),
        StructField("petal_length", FloatType(), False),
        StructField("petal_width", FloatType(), False),
    ])
    df = spark.read.csv("input.csv", schema=schema)

Create a BentoService object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a BentoService object using the BentoML service you want to use for the batch inference
job. Here, we first try to use ``bentoml.get`` to get the bento from the local BentoML store. If it
is not found, we retrieve the bento from the BentoML public S3 and import it.

.. code-block:: python

    import bentoml

    bento = bentoml.get("iris_classifier:latest")

Run the batch inference job
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the batch inference job using the ``bentoml.batch.run_in_spark()`` method. This method takes
the API name, the Spark DataFrame containing the input data, and the Spark session itself as
parameters, and it returns a DataFrame containing the results of the batch inference job.

.. code-block:: python

    results_df = bentoml.batch.run_in_spark(bento, "classify", df, spark)

Internally, what happens when you run ``run_in_spark`` is as follows:

* First, the bento is distributed to the cluster. Note that if the bento has already been
  distributed, i.e. you have already run a computation with that bento, this step is skipped.

* Next, a process function is created, which runs the API method on every Spark batch given it. The
  batch size can be controlled by setting ``spark.sql.execution.arrow.maxRecordsPerBatch``. PySpark
  pickles this process function and dispatches it, along with the relevant data, to the workers.

* Finally, the function is evaluated on the given dataframe. Once all methods that the user defined
  in the script have been executed, the data is returned to the master node.

Save the results
^^^^^^^^^^^^^^^^

Finally, save the results of the batch inference job to a file using the
``DataFrame.write.csv()`` method. This method takes a file path as input and saves the contents
of the DataFrame to the specified file.

.. code-block:: python

    results_df.write.csv("output")

Upon success, you should see multiple files in the output folder: an empty ``_SUCCESS`` file and
one or more ``part-*.csv`` files containing your output.

.. code-block:: bash

    $ ls output
    _SUCCESS  part-00000-85fe41df-4005-4991-a6ad-98b6ed549993-c000.csv
    $ head output/part-00000-d8fe59de-0233-4a80-8bda-519ce98223ea-c000.csv
    1.0
    0.0
    2.0
    0.0

Spark supports many formats other than CSV; see the `Spark documentation
<https://spark.apache.org/docs/latest/api/python//reference/pyspark.sql/api/pyspark.sql.DataFrameWriter.html#pyspark.sql.DataFrameWriter>`_
for a full list.
