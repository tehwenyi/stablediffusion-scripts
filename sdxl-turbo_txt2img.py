import argparse
import math
import torch

from diffusers import AutoPipelineForText2Image
from utils.utils import log, save_images_grid, save_image, create_timestamp, create_folder, start_timer, stop_timer

def initialize_model():
    """Initialize base and refiner models."""
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        print("CUDA is not available. Running on CPU.")

    return pipe

def generate_single_image(pipe, args):
    """Generate a single image based on the provided arguments."""
    # SDXL-Turbo does not make use of guidance_scale or negative_prompt, we disable it with guidance_scale=0.0.
    image = pipe(prompt=args.prompt, 
                 num_inference_steps=args.n_steps, 
                 guidance_scale=0.0).images[0]

    return image

def main(args):
    """Main function for image generation."""
    output_folder = create_folder(args.output_path)
    timestamp = create_timestamp()

    start_time = start_timer()

    pipe = initialize_model()

    generated_images = []
    zero_fill_width = max(1, math.ceil(math.log10(args.num_samples + 1)))
    for idx in range(args.num_samples):
        image = generate_single_image(pipe, args)
        generated_images.append(image)
        save_image(image=image, idx=idx, output_folder=output_folder, timestamp=timestamp, zero_fill_width=zero_fill_width)

    print(f"{idx + 1} images saved in {output_folder}")
    
    grid_output_path = save_images_grid(generated_images, f"Prompt: {args.prompt}", output_folder, timestamp)

    generation_time = stop_timer(start_time)

    log(timestamp=timestamp, 
        args=args, 
        log_filepath=args.log_path, 
        n_images=idx + 1, 
        output_folder=output_folder, 
        grid_output_path=grid_output_path,
        elapsed_time=generation_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from a text prompt using SDXL Turbo.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    # parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for image generation (default: '')")
    parser.add_argument("--n-steps", type=int, default=1, help="Number of steps for image generation (default: 1)")
    # parser.add_argument("--high-noise-frac", type=float, default=0.8, help="High noise fraction for image generation (default: 0.8)")
    parser.add_argument("--output-path", type=str, default="outputs/sdxl-turbo/txt2img/", help="Output path for saving generated images (default: 'outputs/sdxl-turbo/txt2img/')")
    parser.add_argument("--log-path", type=str, default="logs/sdxl-turbo_txt2img_log.txt", help="Output path for the log file (default: 'logs/sdxl-turbo_txt2img_log.txt')")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate (default: 1)")
    # parser.add_argument("--base-only", action="store_true", help="Only generate image through the base (default: False)")

    args = parser.parse_args()
    main(args)
