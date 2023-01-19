========================
Batch Inference in Spark
========================

`Apache Spark <https://spark.apache.org/>`_ is a general-purpose distributed processing system used for big data workloads. It allows for processing large datasets through a unique in-memory computation model, which can improve the performance of big data processing tasks. It also provides a wide range of APIs and a feature-rich set of tools for structured data processing, machine learning, and stream processing for big-data applications.

BentoML now supports running your Bentos with batch data via Spark.

The following tutorial assumes basic understanding of BentoML and a BentoML service ready to use. If you'd like to learn more about BentoML, see the :ref:`BentoML tutorial <tutorial>`.

Make sure to have at least BentoML 1.0.13 and Spark version 3.3.0 available in your system.

.. code-block:: bash

	$ pip install -U "bentoml>=1.0.13"

For this example, we'll be using the quickstart bento from the aforementioned tutorial, but the
commands should work for bentos with IO descriptors which support batch inference (at the time of
writing, those are ``bentoml.io.NumpyNdarray``, ``bentoml.io.PandasDataFrame`` and
``bentoml.io.PandasSeries``) with the following caveat:

IMPORTANT: your Bento API must be capable of accepting multiple inputs. For example,
``batch_classify(np.array([[input_1], [input_2]]))`` must work, and return
``np.array([[output_1], [output_2]])``. The quickstart bento supports this pattern because the iris
classifier model it contains does.

#. Create a PySpark SparkSession object. This will be used to create a DataFrame from the input
   data, and to run the batch inference job. If you're running in a notebook with spark already
   (e.g. a VertexAI PySpark notebook or a Databricks Notebook), you can skip this step.

    .. code-block:: python

        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

#. Load the input data into a PySpark DataFrame. If you are using multipart input, or your dataframe
   requires column names, you must also provide a schema for your DataFrame as you load it. You can
   do this using the ``spark.read.csv()`` method, which takes a file path as input and returns a
   DataFrame containing the data from the file.

    .. code-block:: python
        from pyspark.sql.types import StructType, StructField, FloatType, StringType
        schema = SparkStruct(
            StructField(name=”sepal_length”, FloatType(), False),
            StructField(name=”sepal_width”, FloatType(), False),
            StructField(name=”petal_length”, FloatType(), False),
            StructField(name=”petal_width”, FloatType(), False),
        )
        df = spark.read.csv("https://docs.bentoml.org/en/latest/integrations/spark/input.csv")

#. Create a BentoService object using the BentoML service you want to use for the batch inference
   job. You can do this by calling the ``bentoml.get`` function, and passing the name of the bento
   and its version as a parameter.

    .. code-block:: python

        import bentoml

        bento = bentoml.import_bento("s3://bentoml/quickstart")
        # alternatively, if the bento is already in the bento store:
        bento = bentoml.get("iris_classifier:latest")

#. Run the batch inference job using the ``bentoml.batch.run_in_spark()`` method. This method takes
   the API name, the Spark DataFrame containing the input data, and the Spark session itself as
   parameters, and it returns a DataFrame containing the results of the batch inference job.

    .. code-block:: python

        results_df = bentoml.batch.run_in_spark(bento, "classify", df, spark)

        Internally, what happens when you run `run_in_spark` is as follows:

    * First, the bento is distributed to the cluster. Note that if the bento has already been
      distributed, i.e. you have already run a computation with that bento, this step is skipped.

    * Next, a process function is created, which starts a BentoML server on each of the Spark
      workers, then uses a client to process all the data. This is done so that the workers take
      advantage of the batch processing features of the BentoML server. PySpark pickles this process
      function and dispatches it, along with the relevant data, to the workers.

    * Finally, the function is evaluated on the given dataframe. Once all methods that the user
      defined in the script have been executed, the data is returned to the master node.

#. Finally, save the results of the batch inference job to a file using the
   ``DataFrame.write.csv()`` method. This method takes a file path as input and saves the contents
   of the DataFrame to the specified file.

    .. code-block:: python

        results_df.write.csv("output")

    Upon success, you should see multiple files in the output folder: an empty ``_SUCCESS`` file and
    one or more ``part-*.csv`` files containing your output.

    .. code-block:: bash

        $ ls output
        _SUCCESS  part-00000-85fe41df-4005-4991-a6ad-98b6ed549993-c000.csv

    Spark supports many formats other than CSV; see `the Spark documentation
    <https://spark.apache.org/docs/latest/api/python//reference/pyspark.sql/api/pyspark.sql.DataFrameWriter.html#pyspark.sql.DataFrameWriter>`
    for a full list.