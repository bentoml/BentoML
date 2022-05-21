from pickle_model import PickleModel

import bentoml.picklable_model


def train():
    bentoml.picklable_model.save_model(
        "py_model",
        PickleModel(),
        signatures={
            "predict_file": {"batchable": True},
            "echo_json": {"batchable": True},
            "echo_multi_ndarray": {"batchable": True},
            "predict_ndarray": {"batchable": True},
            "predict_multi_ndarray": {"batchable": True},
            "predict_dataframe": {"batchable": True},
        },
    )


if __name__ == "__main__":
    train()
