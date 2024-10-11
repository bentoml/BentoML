# Model Load and Store with Multiple Models

In this example, we will use LoRA and StableDiffusion to demonstrate how to load and store multiple models both for local models and remote models.

## Install Dependencies
```bash
python -m venv model-load-store-lora
source model-load-store-lora/bin/activate
pip install -r requirements.txt
```

## Import Models

Run the import_model.py script in this example to save models to the BentoML Model Store.
```python
import bentoml
import shutil
import os
from diffusers import StableDiffusionPipeline

model_id = 'CompVis/stable-diffusion-v1-4'
pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

local_lora = "./lora"
with bentoml.models.create(
    name='sd', # All models are created under ~/bentoml/models/{name = sd}
) as model_ref:
    local_diffusion_path = model_ref.path_of("diffusion")
    local_lora_path = model_ref.path_of("lora")
    pipeline.save_pretrained(local_diffusion_path)
    shutil.copytree(local_lora, local_lora_path, dirs_exist_ok=True)
    print(f"Model saved: {model_ref}")
```

When storing multiple models, you can simply create new directories corresponding to these models inside the `bentoml.models.create()` context manager. For local models, use `shutil.copytree()` to copy the local models and files. For remote models, use the respective library tools to save the models to the directory you just created.

To view all the models in the Model Store, run bentoml models list:
```bash
$ bentoml models list
 Tag                                                                             Module                              Size        Creation Time
 sd:l42k7zxcmksxhyln                                                                                                 5.11 GiB    2024-03-15 00:24:22
```

## Load Models
Now that the models are saved, we can get the directory of the saved models by using `model_ref = bentoml.models.get("sd:latest")`. You can get the base path by calling `model_ref.path`. This will return the sd directory you have just created. To
get each individual model subpath, use `model_ref.path_of("{subpath}")`.

```python
class StableDiffusion:
    model_ref = bentoml.models.get("sd:latest")

    def __init__(self) -> None:
        # Load model into pipeline
        self.diffusion_ref = model_ref.path_of("diffusion")
        self.lora_ref = model_ref.path_of("lora")
        self.stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(self.diffusion_ref, use_safetensors=True)
        self.stable_diffusion_txt2img.unet.load_attn_procs(self.lora_ref)
        self.stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
            vae=self.stable_diffusion_txt2img.vae,
            text_encoder=self.stable_diffusion_txt2img.text_encoder,
            tokenizer=self.stable_diffusion_txt2img.tokenizer,
            unet=self.stable_diffusion_txt2img.unet,
            scheduler=self.stable_diffusion_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.stable_diffusion_txt2img.to('cuda')
        self.stable_diffusion_img2img.to('cuda')
```

To serve this Service locally, run:
```bash
bentoml serve .
```

## Deploy to BentoCloud
Run the following command to deploy this example to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.
```bash
bentoml deploy .
```
For more information, see [Create Deployments](https://docs.bentoml.com/en/latest/bentocloud/how-tos/create-deployments.html).
