from pickle_model import PickleModel

import bentoml.sklearn


def train():
    bentoml.sklearn.save("sk_model", PickleModel())


if __name__ == "__main__":
    train()
