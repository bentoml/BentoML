from pickle_model import PickleModel

import bentoml.picklable_model


def train():
    bentoml.picklable_model.save("sk_model", PickleModel())


if __name__ == "__main__":
    train()
