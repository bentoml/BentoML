# Serving YOLOv5 model with BentoML 

This project demonstrate how to use pretrained YOLOv5 model from Torch hub, and use
it to build a prediction service in BentoML.

The model used in this example is built upon https://github.com/ultralytics/yolov5

## Before you started

Install required dependencies:

```bash
pip install -r ./requirements.txt
```

Download the model file via `torch.hub.load`:

```python
import torch
torch.hub.load('ultralytics/yolov5', 'yolov5s')
```

Now you should have a `yolov5s.pt` file created in current directory.

## Run the service

Launch the service locally:

```bash
bentoml serve service.py:svc
```


## Test the endpoint

Visit http://127.0.0.1:3000 to submit input images via the web UI.

The sample service provides two different endpoints:
* `/invocation` - takes an image input and returns predictions in a tabular data format
* `/render` - takes an image input and returns the input image with boxes and labels rendered on top


To test the `/invocation` endpoint:

```bash
curl -X 'POST' \
  'http://localhost:3000/invocation' \
  -H 'accept: image/jpeg' \
  -H 'Content-Type: image/jpeg' \
  --data-binary '@data/bus.jpg'
```

Sample result:
```
[{"xmin":51.4846191406,"ymin":399.7782592773,"xmax":225.7452697754,"ymax":894.1701049805,"confidence":0.8960712552,"class":0,"name":"person"},{"xmin":25.1442546844,"ymin":230.5268707275,"xmax":803.268371582,"ymax":767.0746459961,"confidence":0.8453037143,"class":5,"name":"bus"},{"xmin":219.5371704102,"ymin":398.213684082,"xmax":350.1277770996,"ymax":861.6119384766,"confidence":0.7823933363,"class":0,"name":"person"},{"xmin":671.8472290039,"ymin":432.5200195312,"xmax":810.0,"ymax":877.744934082,"confidence":0.6512392759,"class":0,"name":"person"}]%
```


Test the `/render` endpoint to receive an image with boxes and labels:
```bash
curl -X 'POST' \
  'http://localhost:3000/render' \
  -H 'accept: image/jpeg' \
  -H 'Content-Type: image/jpeg' \
  --data-binary '@data/bus.jpg' \
  --output './output.jpeg'
```

Sample result:

![output-4](https://user-images.githubusercontent.com/489344/178635310-99dc7fde-5224-4fab-84cf-1a87277a0450.jpeg)



## Build Bento

The `bentofile.yaml` have configured all required system packages and python dependencies. 

```bash
bentoml build
```

Once the Bento is built, containerize it as a Docker image for deployment:

```bash
bentoml containerize yolo_v5_demo:latest
```
