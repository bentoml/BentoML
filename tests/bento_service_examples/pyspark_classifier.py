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
        # Convert Pandas to Spark DataFrame (w/required 'features' column)
        # (SparkDataFrameAdapter could do this instead)
        spark_df = spark_session.createDataFrame(pandas_df)
        assembler = VectorAssembler(inputCols=list(pandas_df.columns),
                                    outputCol='features')
        spark_df = assembler.transform(spark_df).select(['features'])

        output_df = self.artifacts.model.transform(spark_df)
        return output_df.select("prediction").toPandas()
