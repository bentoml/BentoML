import pathlib
import sys

from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.sklearn import SklearnModelArtifact


class PickleModel:
    def predict_dataframe(self, df):
        return df["col1"] * 2

    def predict_image(self, input_datas):
        return [input_data.shape for input_data in input_datas]

    def predict_file(self, input_files):
        return [f.read() for f in input_files]

    def predict_multi_images(self, originals, compareds):
        import numpy as np

        eq = np.array(originals) == np.array(compareds)
        return eq.all(axis=tuple(range(1, len(eq.shape))))

    def predict_json(self, input_datas):
        return input_datas


def pack_models(path):
    model = PickleModel()
    PickleArtifact("model").pack(model).save(path)

    from sklearn.ensemble import RandomForestRegressor

    sklearn_model = RandomForestRegressor(n_estimators=2)
    sklearn_model.fit(
        [[i] for _ in range(100) for i in range(10)],
        [i for _ in range(100) for i in range(10)],
    )
    SklearnModelArtifact("sk_model").pack(sklearn_model).save(path)


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    pathlib.Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    pack_models(artifacts_path)
