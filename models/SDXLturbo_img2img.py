import torch

from diffusers import AutoPipelineForImage2Image
from .base_image_generator import BaseImageGenerator

class SDXLTurboImg2Img(BaseImageGenerator):
    def initialize_model(self):
        """Initialize img2img model."""
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16")

        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            print("CUDA is not available. Running on CPU.")

        return pipe, None

    def generate_images(self, base, refiner):
        """Generate images based on the provided arguments."""
        assert refiner is None, "The refiner argument must be None for generate_images function."

        try:
            assert self.init_image is not None
        except AttributeError:
            error_message = (
                "Failed to initialize the image for generation.\n"
                f"Ensure that --init-image-path is provided and points to a valid image file.\n"
                "Additionally, check if the file format is supported."
            )
            raise ValueError(error_message)

        images = base(
            self.prompt, 
            image=self.init_image, 
            num_inference_steps=self.n_steps, 
            strength=self.strength, 
            guidance_scale=0.0,
            num_images_per_prompt=self.num_samples
        ).images

        return images
    