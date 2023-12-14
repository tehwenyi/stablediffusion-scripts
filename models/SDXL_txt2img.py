import torch
from pathlib import Path

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from .base_image_generator import BaseImageGenerator

class SDXLTxt2Img(BaseImageGenerator):
    def initialize_model(self):
        """Initialize base and refiner models."""
        base = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )

        if self.lora_weights:
            lora_path = Path(self.lora_weights)
            lora_folder = str(lora_path.parent)
            lora_filename = str(lora_path.name)
            base.load_lora_weights(lora_folder, weight_name=lora_filename)
        
        refiner = None
        if not self.base_only:
            refiner = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        if torch.cuda.is_available():
            base.to("cuda")
            if refiner is not None:
                refiner.to("cuda")
        else:
            print("CUDA is not available. Running on CPU.")

        return base, refiner

    def generate_images(self, base, refiner):
        """Generate images based on the provided arguments."""

        if self.base_only:
            images = base(prompt=self.prompt).images

        else:
            base_output = base(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.n_steps,
                # denoising_end=self.high_noise_frac,
                output_type="latent",
                num_images_per_prompt=self.num_samples
            ).images

            images = refiner(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.n_steps,
                # denoising_start=self.high_noise_frac,
                image=base_output,
                num_images_per_prompt=self.num_samples
            ).images

        return images
