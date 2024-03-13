import os
import typing as t

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

import bentoml

sample_txt2img_input = dict(
    prompt="pokemon with blue eyes",
    height=768,
    width=768,
    num_inference_steps=50,
    guidance_scale=7.5,
    eta=0.0,
)

sample_img2img_input = dict(prompt="make the image black and white", strength=0.8)


@bentoml.service(
    resources={"memory": "500MiB"},
    traffic={"timeout": 30},
)
class StableDiffusion:
    model_ref = bentoml.models.get("sd2:latest")

    def __init__(self) -> None:
        # Load model into pipeline
        self.diffusion_ref = os.path.join(self.model_ref.path, "diffusion")
        self.lora_ref = os.path.join(self.model_ref.path, "lora")
        self.stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(
            self.diffusion_ref, use_safetensors=True
        )
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
        self.stable_diffusion_txt2img.to("cuda")
        self.stable_diffusion_img2img.to("cuda")

    @bentoml.api
    def txt2img(self, input_data: t.Dict[str, t.Any] = sample_txt2img_input) -> Image:
        res = self.stable_diffusion_txt2img(**input_data)
        image = res[0][0]
        return image

    @bentoml.api
    def img2img(
        self, image: Image, input_data: t.Dict[str, t.Any] = sample_img2img_input
    ) -> Image:
        image = image
        res = self.stable_diffusion_img2img(image=image, **input_data)
        image = res[0][0]
        return image
