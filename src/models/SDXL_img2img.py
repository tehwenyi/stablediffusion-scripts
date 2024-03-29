import torch

from diffusers import AutoPipelineForImage2Image
from .base_image_generator import BaseImageGenerator

class SDXLImg2Img(BaseImageGenerator):
    def initialize_model(self):
        """Initialize img2img model."""
        args = {"pretrained_model_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0", "use_safetensors": True}
        
        if self.use_fp16:
            args.update({"torch_dtype": torch.float16, "variant": "fp16"})

        if self.local_weights_path:
                args.update({"pretrained_model_or_path": self.local_weights_path[0]})

        pipe = AutoPipelineForImage2Image.from_pretrained(**args)

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
            negative_prompt=self.negative_prompt,
            image=self.init_image,
            strength=self.strength,
            denoising_end=self.high_noise_frac,
            num_inference_steps=self.n_steps,
            num_images_per_prompt=self.num_samples
        ).images

        return images
