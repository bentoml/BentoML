import bentoml
import shutil
import os
from diffusers import StableDiffusionPipeline

model_id = 'CompVis/stable-diffusion-v1-4'
pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

local_lora = "./lora"
with bentoml.models.create(
    name='sd2',
) as model_ref:
    local_diffusion_path = model_ref.path + '/diffusion'
    local_lora_path = model_ref.path + '/lora'
    os.mkdir(local_diffusion_path)
    os.mkdir(local_lora_path)
    pipeline.save_pretrained(local_diffusion_path)
    shutil.copytree(local_lora, local_lora_path, dirs_exist_ok=True)
    print(f"Model saved: {model_ref}")
