import torch
from PIL.Image import Image

import bentoml


@bentoml.service(
    resources={
        "memory": "500MiB",
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4",
    },
    traffic={"timeout": 600},
)
class StableDiffusion:
    model_ref = bentoml.models.get("sd:latest")

    def __init__(self) -> None:
        from diffusers import StableDiffusionPipeline

        # Load model into pipeline
        self.diffusion_ref = self.model_ref.path_of("diffusion")
        self.lora_ref = self.model_ref.path_of("lora")
        self.stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(
            self.diffusion_ref, use_safetensors=True
        )
        self.stable_diffusion_txt2img.unet.load_attn_procs(self.lora_ref)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stable_diffusion_txt2img.to(device)

    @bentoml.api
    def txt2img(
        self,
        prompt: str = "A Pokemon with blue eyes.",
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        eta: int = 0,
    ) -> Image:
        input_data = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        )
        res = self.stable_diffusion_txt2img(**input_data)
        image = res[0][0]
        return image
