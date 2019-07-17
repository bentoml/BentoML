
import bentoml
from bentoml import api, artifacts, env, BentoService
from bentoml.artifact import KerasModelArtifact, PickleArtifact
from bentoml.handlers import ImageHandler
from tensorflow.image import resize
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@bentoml.env(conda_dependencies=['tensorflow', 'numpy'])
@bentoml.artifacts([KerasModelArtifact('classifier')])
class KerasFashionMnistService(bentoml.BentoService):
        
    @bentoml.api(ImageHandler, pilmode='L')
    def predict(self, image_array):
        if image_array.shape != (28, 28, 1):
            image_array = resize(image_array, (28, 28))
        image_array = image_array.reshape(1, 28, 28, 1)
        
        result = self.artifacts.classifier.predict_classes(image_array)[0]
        return class_names[result]
