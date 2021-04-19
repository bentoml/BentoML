import pathlib
import sys

import paddle
from paddle import nn
from paddle.static import InputSpec
from bentoml.frameworks.paddle import PaddlePaddleModelArtifact


BATCH_SIZE = 8
BATCH_NUM = 4
EPOCH_NUM = 5

IN_FEATURES = 13
OUT_FEATURES = 1


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[IN_FEATURES], dtype='float32')])
    def forward(self, x):
        return self.fc(x)


def train(model, loader, loss_fn, opt):
    model.train()
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (feature, label) in enumerate(loader()):
            out = model(feature)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


def pack_models(path):
    model = Model()
    loss = nn.MSELoss()
    adam = paddle.optimizer.Adam(parameters=model.parameters())

    train_data = paddle.text.datasets.UCIHousing(mode="train")

    loader = paddle.io.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
    )

    train(model, loader, loss, adam)

    PaddlePaddleModelArtifact("model").pack(model).save(path)


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    pathlib.Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    pack_models(artifacts_path)
