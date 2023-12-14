import torch
from pathlib import Path

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from .base_image_generator import BaseImageGenerator

class SDXLTxt2Img(BaseImageGenerator):
    def initialize_model(self):
        """Initialize base and refiner models."""
        # base_args = {"pretrained_model_or_path": "stabilityai/stable-diffusion-xl-base-1.0", "use_safetensors": True}
        base_args = {"pretrained_model_or_path": "/models/stable-diffusion-xl-base-1.0", "use_safetensors": True}
        refiner_args = {"pretrained_model_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0", "use_safetensors": True}
        
        if self.use_fp16:
            base_args.update({"torch_dtype": torch.float16, "variant": "fp16"})
            refiner_args.update({"torch_dtype": torch.float16, "variant": "fp16"})
        
        base = AutoPipelineForText2Image.from_pretrained(**base_args)

        if self.lora_weights:
            lora_path = Path(self.lora_weights)
            lora_folder = str(lora_path.parent)
            lora_filename = str(lora_path.name)
            print(f"Loading lora weights {lora_folder} {lora_filename}")
            base.load_lora_weights(lora_folder, weight_name=lora_filename)

        refiner = None
        if not self.base_only:
            refiner = AutoPipelineForImage2Image.from_pretrained(
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                **refiner_args
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
