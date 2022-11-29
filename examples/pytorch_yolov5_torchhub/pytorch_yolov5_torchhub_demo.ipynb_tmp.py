# %%
"""
# BentoML PyTorch YoloV5 Tutorial

Link to source code: https://github.com/bentoml/BentoML/tree/main/examples/pytorch_yolov5_torchhub/

Here is a quick example of how to use BentoML to serve a TorchHub model. In this example, we will be using the YoloV5 model from TorchHub. We will be using the `torch.hub.load` API to load the model from TorchHub. 
We will then use BentoML to create a REST API server for the model. We will also create a simple web app to interact with the model.

Take `ultralytics/yolov5` as the example.
"""

# %%
!pip install -r requirements.txt

# %%
"""
## Step 1. Copy the source repository to your project directory

We need two things from the source repository:
    * the dependencies, typically in a `requirements.txt` file
    * the source code of the PyTorch Module

Due to the limitation of PyTorch serialization, the source code of the model is not saved in the saved bundle. 
Therefore, we always need to the source code when loading the model from the saved bundle.

Let's clone the source code from the original repo.
"""

# %%
!git clone 'https://github.com/ultralytics/yolov5.git'

# %%
"""
Then we copy the requirements.txt file to the project directory, and append the `pydantic` package to the file.
"""

# %%
!cp yolov5/requirements.txt ./requirements.txt
# !echo "pydantic" >> requirements.txt

# %%
"""
## Step 2. Load the pre-trained model and do some fine-tuning.
"""

# %%
import torch

# Model
original_model = torch.hub.load('./yolov5', 'yolov5s', pretrained=True, source="local")
# Do your custom fine-tune...
# For more details, see https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data


# %%
"""
## Step 3. Normalize the input and output of the model

The output of the yolov5 model is a tensor that mixed the detected objects, position box and probability like:

category_id, x, y, width, height, probability

We need to extract the outputs to a list of dict, each dict contains the information of a detected object before saving the model with bentoml.
"""

# %%
class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, imgs):
        outputs = self.model(imgs)

        # convert outputs to a json serializable list
        results = []
        for det in outputs.pred:
            detections = []
            for i in det:
                d = {}
                d['obj'] = outputs.names[int(i[5])]
                d['position'] = i[:4].tolist()
                d['prob'] = i[4].tolist()
                detections.append(d)
            results.append(detections)

        return results


model = WrapperModel(original_model)


# %%
"""
Now we can save the model with bentoml.
"""

# %%
import bentoml

bentoml.pytorch.save_model(
    "pytorch_yolov5",
    model,
    signatures={"__call__": {"batchable": True, "batchdim": 0}},
)

# %%
"""
## Step 4. Create a BentoML Service for serving the model

Tips:
* using `%%writefile` here because `bentoml.Service` instance must be created in a separate `.py` file
* Even though we have only one model, we can create as many api endpoints as we want.
"""

# %%
%%writefile service.py

import os
import sys
import typing as t

import numpy as np
import PIL.Image

import bentoml
from bentoml.io import JSON
from bentoml.io import Image

yolo_runner = bentoml.pytorch.get("pytorch_yolov5").to_runner()

svc = bentoml.Service(
    name="pytorch_yolo_demo",
    runners=[yolo_runner],
)


sys.path.append('yolov5')

@svc.api(input=Image(), output=JSON())
def predict_image(img: PIL.Image.Image) -> list:
    assert isinstance(img, PIL.Image.Image)
    return yolo_runner.run([np.array(img)])

# %%
"""
Start a dev model server to test out the service defined above
"""

# %%
!bentoml serve service.py:svc

# %%
"""
Now you can use something like:

`curl -H "Content-Type: multipart/form-data" -F 'fileobj=@yolov5/data/images/bus.jpg;type=image/png' http://127.0.0.1:3000/predict_image`
    
to send an image to the digit recognition service
"""

# %%
"""
## Step 5. Build a Bento for distribution and deployment
"""

# %%
"""
Starting a dev server with the Bento build:
"""

# %%
!bentoml build

# %%
"""
## Step 6. Deploy the Bento to production

BentoML supports deploying the BentoML service to AWS Lambda, AWS SageMaker, GCP Cloud Run, Azure Functions, Kubernetes, any docker-compatible environment or to a local server.
"""

# %%
# to deploy to local server:
!bentoml serve pytorch_yolo_demo:latest --production

# deploy to docker:
!bentoml containerize pytorch_yolo_demo -t pytorch_yolo_demo:latest
!docker run -p 5000:5000 pytorch_yolo_demo:latest

# to deploy to AWS Lambda:
!bentoctl deploy aws-lambda pytorch_yolo_demo:latest
