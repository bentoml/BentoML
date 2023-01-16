from __future__ import annotations

import os
import argparse
import tempfile

import onnx
import torch
import torch.nn as nn
import torchvision
import onnx.checker
import torch.nn.functional as F
import torchvision.transforms as transforms

import bentoml


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = out.view(out.size(0), -1)
        out = out.flatten(start_dim=1)
        logits = self.layer3(out)
        out = F.softmax(logits, dim=1)
        return out


## utility function to compute accuracy
def get_accuracy(output, target, batch_size):
    """Obtain accuracy for training round"""
    corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--opset-version", type=int, default=14)

    args = parser.parse_args()

    ## transformations
    ## (Lambda function is to make digits black on white instead of white on black)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: 1 - x)]
    )

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    ## download and load training dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    ## download and load testing dataset
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## train the model
    for epoch in range(args.epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        ## commence training
        model.train()

        ## training step
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            ## forward + backprop + loss
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(predictions, labels, args.batch_size)

        model.eval()
        print(
            "Epoch: %d | Loss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / i, train_acc / i)
        )

    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader, 0):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_acc += get_accuracy(outputs, labels, args.batch_size)

    print("Test Accuracy: %.2f" % (test_acc / i))

    path = tempfile.mkdtemp(suffix="mnist_onnx")
    model_path = os.path.join(path, "mnist.onnx")

    torch.onnx.export(
        model,
        torch.randn(1, 1, 28, 28).to(device),
        model_path,
        verbose=True,
        export_params=True,
        input_names=["input_0"],
        output_names=["output_0"],
        opset_version=args.opset_version,
        dynamic_axes={"input_0": {0: "batch_size"}, "output_0": {0: "batch_size"}},
    )

    ModelProto = onnx.load(model_path)
    onnx.checker.check_model(ModelProto)

    _ = bentoml.onnx.save_model(
        "onnx-mnist",
        ModelProto,
        signatures={"run": {"batchable": True}},
    )
    print("Saved model:", _)


if __name__ == "__main__":
    try:
        m = bentoml.onnx.get("onnx-mnist")
        print("Model exists:", m)
    except bentoml.exceptions.NotFound:
        main()
