import bentoml
from bentoml.io import Image
import bentoml.transformers

runner = bentoml.transformers.load_runner(tag, tasks='image-classification',device=-1,feature_extractor="google/vit-large-patch16-224")

svc = bentoml.Service("vit-object-detection")

@svc.api(input=Image(), output=Image())
def predict(input_img):
    return input_img

app = svc.asgi_app
