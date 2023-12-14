import math

from utils.utils import log_generation_info, save_images_and_prompt_grid, save_images, create_timestamp, create_folder, load_image_from_path, start_timer, stop_timer

class BaseImageGenerator:
    def __init__(self, prompt, negative_prompt, n_steps, init_image_path, output_path, strength, log_path, num_samples, high_noise_frac, use_fp16=False, local_weights_path=None, lora_weights=None, base_only=None):
        # self.model_name = model_name
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.n_steps = n_steps
        self.init_image_path = init_image_path
        try:
            if self.init_image_path: self.init_image = load_image_from_path(self.init_image_path)
            else: self.init_image = None
        except FileNotFoundError as e:
            print(f"[init_image_path] File not found: {self.init_image_path}")
        except PermissionError as e:
            print(f"[init_image_path] Permission error: {e}")
        except Exception as e:
            print(f"[init_image_path] An error occurred while loading the image: {e}")
        self.output_folder = create_folder(output_path)
        self.strength = strength
        self.log_path = log_path
        self.num_samples = num_samples
        self.timestamp = create_timestamp()
        self.grid_image_path = self.output_folder / (self.timestamp + "grid.png")
        self.high_noise_frac = high_noise_frac
        self.use_fp16 = use_fp16
        self.local_weights_path = local_weights_path
        self.lora_weights = lora_weights
        self.base_only = base_only

    def initialize_model(self):
        raise NotImplementedError("Subclasses must implement the initialize_model method.")

    def generate_images(self, base, refiner):
        raise NotImplementedError("Subclasses must implement the generate_images method.")

    def main(self):
        """Main function for image generation."""
        start_time = start_timer()

        base, refiner = self.initialize_model()

        zero_fill_width = max(1, math.ceil(math.log10(self.num_samples + 1)))
        generated_images = self.generate_images(base, refiner)
        save_images(images=generated_images, output_folder=self.output_folder, timestamp=self.timestamp, zero_fill_width=zero_fill_width)

        save_images_and_prompt_grid(images=generated_images, prompt=f"Prompt: {self.prompt}", output_path=self.grid_image_path)

        generation_time = stop_timer(start_time)

        log_generation_info(args=self.__dict__, n_images=len(generated_images), elapsed_time=generation_time)
