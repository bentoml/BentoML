# Model Load and Store with Multiple Models

In this example, we will use LoRA and StableDiffusion to demonstrate how to load and store multiple models both for local models and remote models. 

## Utilization
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## Model Store (import_model.py)

```python
import bentoml
import shutil
import os
from diffusers import StableDiffusionPipeline

model_id = 'CompVis/stable-diffusion-v1-4'
pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

local_lora = "./lora"
with bentoml.models.create(
    name='sd2', # All models are created under ~/bentoml/models/{name = sd2}
) as model_ref:
    local_diffusion_path = os.path.join(model_ref.path, "diffusion")
    local_lora_path = os.path.join(model_ref.path, "lora")
    os.mkdir(local_diffusion_path)
    os.mkdir(local_lora_path)
    pipeline.save_pretrained(local_diffusion_path)
    shutil.copytree(local_lora, local_lora_path, dirs_exist_ok=True)
    print(f"Model saved: {model_ref}")
```

When storing multiple models, you can simply create new directories corresponding to these models inside the bentoml.models.create() context manager. For local models, use shutil.copytree to copy the local models and files. For remote models, use the respective library tools to save the models to the directory you just created.

## Model Load (service.py)
Now that the models are saved, we can get the directory of the saved models by using `model_ref = bentoml.models.get("sd2:latest")`. You can get the path by calling `model_ref.path`. This will return the sd2 directory you have just created. To 
get each individual model, simply append the directory names ('diffusion and 'lora' in this example). 

```python
class StableDiffusion:
    model_ref = bentoml.models.get("sd2:latest")
    
    def __init__(self) -> None:
        # Load model into pipeline
        self.diffusion_ref = os.path.join(self.model_ref.path, "diffusion")
        self.lora_ref = os.path.join(self.model_ref.path, "lora")
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
