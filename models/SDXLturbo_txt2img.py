import torch

from diffusers import AutoPipelineForText2Image
from .base_image_generator import BaseImageGenerator

class SDXLTurboTxt2Img(BaseImageGenerator):
    def initialize_model(self):
        """Initialize base and refiner models."""
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            print("CUDA is not available. Running on CPU.")

        return pipe, None

    def generate_images(self, base, refiner):
        """Generate images based on the provided arguments."""

        print("Note: SDXL-Turbo does not make use of guidance_scale or negative_prompt, we disable it with guidance_scale=0.0")

        images = base(
            prompt=self.prompt, 
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            guidance_scale=0.0,
            num_images_per_prompt=self.num_samples
        ).images

        return images
