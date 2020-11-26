import pytest
import urllib
import numpy as np
import cv2
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.detectron2_classifier import DetectronClassifier
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


@pytest.fixture()
def detectron2_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    DetectronClassifier._bento_service_bundle_path = None
    DetectronClassifier._bento_service_bundle_version = None
    return DetectronClassifier

def test_detectron2_artifact_pack(detectron2_classifier_class):

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    clone_cfg = cfg.clone()  # cfg can be modified by model
    model = build_model(clone_cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    req = urllib.request.urlopen('http://images.cocodataset.org/val2017/000000439715.jpg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    svc = detectron2_classifier_class()
    svc.pack('model', model)
    response = svc.predict(image)
    assert response['scores'][0] > 0.9
    comparison = np.array(response['classes']) == np.array([17, 0, 0, 0, 0, 0, 0, 0, 25, 0, 25, 25, 0, 0, 24])
    assert comparison.all()

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
