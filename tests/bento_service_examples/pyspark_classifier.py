import bentoml
from bentoml.adapters import DataframeInput
from bentoml.artifact import PysparkModelArtifact

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
spark_session = SparkSession.builder.appName('BentoService').getOrCreate()


@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([PysparkModelArtifact('model')])
class PysparkClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, pandas_df):
        # Pandas DF -> Spark DF -> Spark DF with "features" Vector column
        spark_df = spark_session.createDataFrame(pandas_df)
        column_labels = [str(c) for c in list(pandas_df.columns)]
        assembler = VectorAssembler(inputCols=column_labels,
                                    outputCol='features')
        spark_df = assembler.transform(spark_df).select(['features'])

        # Run inference
        output_df = self.artifacts.model.transform(spark_df)

        # Spark DF -> Pandas DF -> Numpy Array
        pred = output_df.select("prediction").toPandas().prediction.values

        return pred
