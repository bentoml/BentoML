from pyspark.sql import SparkSession
import pyspark
import bentoml

spark = SparkSession.builder.master("local[1]").appName("testApp").getOrCreate()

df = spark.read.csv("data.csv")

df.printSchema()

import bentoml._internal.spark

udf = bentoml._internal.spark.get_udf(spark, "batch_processor:latest", "classify1")

udf(df)