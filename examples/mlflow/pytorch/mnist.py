# pylint: disable=redefined-outer-name
#
# Trains an MNIST digit recognizer using PyTorch, and uses tensorboardX to log training metrics
# and weights in TensorBoard event format to the MLflow run's artifact directory. This stores the
# TensorBoard events in MLflow for later access using the TensorBoard command line tool.
#
# NOTE: This example requires you to first install PyTorch (using the instructions at pytorch.org)
#       and tensorboardX (using pip install tensorboardX).
#
# Code based on https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/mnist/main.py.
#
# BentoML example is based on https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py
#
import os
import argparse

import torch
import mlflow
import torch.nn as nn
import torch.optim as optim
import mlflow.pytorch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import bentoml

# Command-line arguments
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--enable-cuda",
    type=str,
    choices=["True", "False"],
    default="True",
    help="enables or disables CUDA training",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == "True" else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar("train_loss", loss.data.item(), step)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).data.item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    step = (epoch + 1) * len(train_loader)
    log_scalar("test_loss", test_loss, step)
    log_scalar("test_accuracy", test_accuracy, step)


def log_scalar(name, value, _):
    """Log a scalar value to both MLflow and TensorBoard"""
    mlflow.log_metric(name, value)


if __name__ == "__main__":
    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # Perform the training
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model")
        print(
            "\nThe model is logged at:\n%s"
            % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )

        # Option 1: Save pytorch model directly with BentoML
        bento_model_1 = bentoml.pytorch.save_model(
            "pytorch-mnist", model, signatures={"__call__": {"batchable": True}}
        )
        print("Pytorch Model saved with BentoML: %s" % bento_model_1)

        # make predictions with BentoML runner
        model_runner_1 = bentoml.pytorch.get("pytorch-mnist:latest").to_runner()
        model_runner_1.init_local()

        # Extract a few examples from the test dataset to evaluate on
        eval_data, eval_labels = next(iter(test_loader))

        predictions = model_runner_1.run(eval_data.numpy())
        template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
        print("\nSample predictions")
        for index in range(5):
            print(
                template.format(index, eval_labels[index], predictions.argmax(1)[index])
            )

        # Option 2: Import logged mlflow model to BentoML for serving:
        model_uri = mlflow.get_artifact_uri("pytorch-model")
        bento_model_2 = bentoml.mlflow.import_model(
            "mlflow_pytorch_mnist",
            model_uri,
            signatures={"predict": {"batchable": True}},
        )
        print("Model imported to BentoML: %s" % bento_model_2)

        # make predictions with BentoML runner
        model_runner_2 = bentoml.mlflow.get("mlflow_pytorch_mnist:latest").to_runner()
        model_runner_2.init_local()

        # Extract a few examples from the test dataset to evaluate on
        eval_data, eval_labels = next(iter(test_loader))

        predictions = model_runner_2.predict.run(eval_data.numpy())
        template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
        print("\nSample predictions")
        for index in range(5):
            print(
                template.format(index, eval_labels[index], predictions.argmax(1)[index])
            )
