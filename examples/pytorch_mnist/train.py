# pylint: disable=redefined-outer-name
import os
import random
import argparse

import model as models
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import KFold

import bentoml

K_FOLDS = 5
NUM_EPOCHS = 3
LOSS_FUNCTION = nn.CrossEntropyLoss()


# reproducible setup for testing
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _dataloader_init_fn():
    np.random.seed(seed)


def get_dataset():
    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    train_set = MNIST(
        os.getcwd(), download=True, transform=transforms.ToTensor(), train=True
    )
    test_set = MNIST(
        os.getcwd(), download=True, transform=transforms.ToTensor(), train=False
    )
    return train_set, test_set


def train_epoch(model, optimizer, loss_function, train_loader, epoch, device="cpu"):
    # Mark training flag
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 499 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_model(model, test_loader, device="cpu"):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct, total


def cross_validate(dataset, epochs=NUM_EPOCHS, k_folds=K_FOLDS, device="cpu"):
    results = {}

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    print("--------------------------------")

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            sampler=train_subsampler,
            worker_init_fn=_dataloader_init_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            sampler=test_subsampler,
            worker_init_fn=_dataloader_init_fn,
        )

        # Train this fold
        model = models.SimpleConvNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_function = nn.CrossEntropyLoss()
        model = model.to(device)
        for epoch in range(epochs):
            train_epoch(
                model, optimizer, loss_function, train_loader, epoch, device=device
            )

        # Evaluation for this fold
        correct, total = test_model(model, test_loader, device)
        print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
        print("--------------------------------")
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {K_FOLDS} FOLDS")
    print("--------------------------------")
    sum_ = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        sum_ += value

    print(f"Average: {sum_/len(results.items())} %")

    return results


def train(dataset, epochs=NUM_EPOCHS, device="cpu"):
    print("Training using %s." % device)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        sampler=train_sampler,
        worker_init_fn=_dataloader_init_fn,
    )
    model = models.SimpleConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    model = model.to(device)
    for epoch in range(epochs):
        train_epoch(model, optimizer, loss_function, train_loader, epoch, device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BentoML PyTorch MNIST Example")
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        metavar="N",
        help=f"number of epochs to train (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=K_FOLDS,
        metavar="N",
        help=f"number of folds for k-fold cross-validation (default: {K_FOLDS}, 1 to disable cv)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="enable CUDA training"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pytorch_mnist",
        help="name for saved the model",
    )

    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_set, test_set = get_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=10,
        sampler=torch.utils.data.RandomSampler(test_set),
        worker_init_fn=_dataloader_init_fn,
    )

    if args.k_folds > 1:
        cv_results = cross_validate(train_set, args.epochs, args.k_folds, device)
    else:
        cv_results = {}

    trained_model = train(train_set, args.epochs, device)
    correct, total = test_model(trained_model, test_loader, device)

    # training related
    metadata = {
        "acc": float(correct) / total,
        "cv_stats": cv_results,
    }

    signatures = {"predict": {"batchable": True}}

    saved_model = bentoml.pytorch.save_model(
        args.model_name,
        trained_model,
        signatures=signatures,
        metadata=metadata,
        external_modules=[models],
    )
    print(f"Saved model: {saved_model}")
