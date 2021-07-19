import numpy as np
from easyocr import Reader


def predict_image(model: Reader, image: np.ndarray):
    return [x[1] for x in model.readtext(image)]
