=======
PyTorch
=======

BentoML provides native support for serving and deploying models trained from PyTorch. For more in-depth tutorials about PyTorch, please visit `PyTorch's official documentation <https://pytorch.org/tutorials/>`_

Preface
-------

If you have already compiled your PyTorch model to TorchScript, you should consider using BentoML's first-class module :doc:`bentoml.torchscript </reference/frameworks/torchscript>` instead, as it is less likely to cause compatibility issues during production.

.. note::

    :bdg-info:`Remarks:` We recommend users to apply model optimization techniques such as `distillation <https://arxiv.org/abs/1503.02531>`_ or `quantization <https://pytorch.org/docs/stable/quantization.html#general-quantization-flow>`_ . Alternatively, PyTorch models can also be converted to :doc:`/frameworks/onnx` models and leverage different runtimes (e.g. TensorRT, Apache TVM, etc.) for better performance.


Saving a Trained Model
----------------------

For common PyTorch models with single input:

.. code-block:: python
    :caption: `train.py`

    import bentoml
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # a small epoch just for demostration purpose
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('Epoch: %d, Step: %d, Loss: %.4f' % (epoch, i, loss.item()))

    bentoml.pytorch.save(
        model,
        "my_torch_model",
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )


``bentoml.pytorch`` also supports saving models that take multiple tensors as input:

.. code-block:: python
    :caption: `train.py`

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim


    class Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y


    model = Net()
    ... # training

    bentoml.pytorch.save(
        model,
        "my_torch_model",
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )

.. note::

    :bdg-info:`Remarks:` External python classes or utility functions required by the model must be referenced in ``<module>.<class>`` format, and such modules should be passed to ``bentoml.pytorch.save`` via ``external_modules``. For example:

    .. code-block:: python
       :caption: `train.py`
       :emphasize-lines: 1,8

       import my_models

       model = my_models.MyModel()
       bentoml.pytorch.save(
           model,
           "my_torch_model",
           signatures={"__call__": {"batchable": True, "batch_dim": 0}},
           external_modules=[my_models],
       )

    This is due to a limitation from PyTorch model serialisation, where PyTorch requires the model's source code to restore it.

    A better practice is to compile your model to `TorchScript <https://pytorch.org/docs/stable/jit.html>`_ format.

.. note::

    :code:`bentoml.pytorch.save_model` has parameter ``signatures``.
    The ``signatures`` argument of type :ref:`Model Signatures <concepts/model:Model Signatures>` in :obj:`bentoml.pytorch.save_model` is used to determine which methods will be used for inference and exposed in the Runner. The signatures dictionary will then be used during the creation process of a Runner instance.

The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``. This means by default, BentoML’s `Adaptive Batching <guides/batching:Adaptive Batching>`_ is disabled when using :obj:`~bentoml.pytorch.save_model()`. If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow:



.. code-block:: python

    bentoml.pytorch.save(model, "my_model", signatures={"__call__": {"batch_dim": 0, "batchable": True}})


Building a Service
------------------

Create a BentoML service with the previously saved `my_torch_model` pipeline using the :code:`bentoml.pytorch` framework APIs.

.. code-block:: python
    :caption: `service.py`

    runner = bentoml.pytorch.get("my_torch_model").to_runner()

    svc = bentoml.Service(name="test_service", runners=[runner])

    @svc.api(input=JSON(), output=JSON())
    async def predict(json_obj: JSONSerializable) -> JSONSerializable:
        batch_ret = await runner.async_run([json_obj])
        return batch_ret[0]

.. note::

    Follow the steps to get the best performance out of your PyTorch model.
    #. Apply adaptive batching if possible.
    #. Serve on GPUs if applicable.
    #. See performance guide from `PyTorch Model Opt Doc <https://pytorch.org/tutorials/beginner/profiler.html>`_


Adaptive Batching
-----------------

Most PyTorch models can accept batched data as input. If batched interence is supported, it is recommended to enable batching to take advantage of
the adaptive batching capability to improve the throughput and efficiency of the model. Enable adaptive batching by overriding the :code:`signatures`
argument with the method name and providing :code:`batchable` and :code:`batch_dim` configurations when saving the model to the model store.

.. seealso::

   See :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` to learn more.


.. note::

   You can find more examples for **PyTorch** in our :examples:`bentoml/examples <>` directory.

.. currentmodule:: bentoml.pytorch
