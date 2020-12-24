import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.spark_mllib import PySparkModelArtifact

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark_session = SparkSession.builder.appName("BentoService").getOrCreate()


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([PySparkModelArtifact('model')])
class PysparkClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, pandas_df):
        spark_df = spark_session.createDataFrame(pandas_df)
        column_labels = [str(c) for c in list(pandas_df.columns)]
        assembler = VectorAssembler(inputCols=column_labels, outputCol='features')
        spark_df = assembler.transform(spark_df).select(['features'])

        output_df = self.artifacts.model.transform(spark_df)

        pred = output_df.select("prediction").toPandas().prediction.values

        return pred
