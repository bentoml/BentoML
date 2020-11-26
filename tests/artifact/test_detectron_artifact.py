import pytest
import builtins
from unittest.mock import patch, MagicMock, mock_open
from bentoml.frameworks.detectron import DetectronModelArtifact


def test_pack():
    test_artifact = DetectronModelArtifact("model")
    dummy_object = True
    returned_object = test_artifact.pack(dummy_object)

    assert test_artifact._model == returned_object._model
    assert test_artifact._file_name == returned_object._file_name
    assert test_artifact._aug == returned_object._aug


def test_load():
    test_artifact = DetectronModelArtifact("model")

    class FakeMetaArch:
        def __init__(self):
            pass

        def get(self):
            return self

        def eval(self):
            pass

        def to(self, device):
            self.device = device
        
        def __call__(self, cfg):
            return self
    
    class MODEL:
        META_ARCHITECTURE = "Test"
    
    class INPUT:
        MIN_SIZE_TEST = 100
        MAX_SIZE_TEST = 1000

    class FakeConfig:
        def __init__(self):
            self.MODEL = MODEL
            self.INPUT = INPUT

        def merge_from_file(self, file):
            self.file_name = file

    class FakeDetectionCheckPointer:
        def __init__(self):
            pass

        def load(self, model_path):
            self.model_path = model_path

    fake_meta_arch = FakeMetaArch()
    fake_config = FakeConfig()
    fake_check_pointer = FakeDetectionCheckPointer()

    with patch("detectron2.modeling.META_ARCH_REGISTRY.get", MagicMock(return_value=fake_meta_arch)):
        with patch("detectron2.config.get_cfg", MagicMock(return_value=fake_config)):
            with patch("detectron2.checkpoint.DetectionCheckpointer", 
                    MagicMock(return_value=fake_check_pointer)):
                test_artifact.load("test")

    assert fake_meta_arch.device == "cpu"
    assert fake_config.file_name == "test/model.yaml"
    assert fake_check_pointer.model_path == "test/model.pth"


def test_save():
    test_artifact = DetectronModelArtifact("model")
    dummy_object = True
    returned_object = test_artifact.pack(dummy_object)

    class FakeDetectionCheckPointer:
        def __init__(self):
            pass
        
        def __call__(self, model, dest):
            pass

        def save(self, file_name):
            self.file_name = file_name

    class FakeConfig:
        def __init__(self):
            pass

        def merge_from_file(self, file):
            self.file_name = file
        
        def dump(self):
            return "Test"

    fake_check_pointer = FakeDetectionCheckPointer()
    fake_config = FakeConfig()
    with patch("os.makedirs", MagicMock(return_value=True)):
    
        with patch("detectron2.checkpoint.DetectionCheckpointer", 
                    MagicMock(return_value=fake_check_pointer)):
            with patch("detectron2.config.get_cfg", MagicMock(return_value=fake_config)):
                with patch('builtins.open', mock_open()) as m:
                    test_artifact.save("test")
                    m.assert_called_once_with('test/model.yaml', 'w', encoding='utf-8')
                    handle = m()
                    handle.write.assert_called_once_with("Test")
