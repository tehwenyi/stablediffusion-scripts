import argparse
import math
import torch

from diffusers import AutoPipelineForImage2Image
from utils.utils import log, save_images_grid, save_image, create_timestamp, create_folder, load_image_from_path, start_timer, stop_timer

def initialize_model():
    """Initialize img2img model."""
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.float16, 
        variant="fp16")

    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        print("CUDA is not available. Running on CPU.")

    return pipe

def generate_single_image(pipe, init_image, args):
    """Generate a single image based on the provided arguments."""
    image = pipe(args.prompt, 
                 image=init_image, 
                 num_inference_steps=2, 
                 strength=args.strength, 
                 guidance_scale=0.0).images[0]

    return image

def main(args):
    """Main function for image generation."""
    output_folder = create_folder(args.output_path)
    timestamp = create_timestamp()

    start_time = start_timer()

    pipe = initialize_model()

    init_image = load_image_from_path(args.init_image_path)

    generated_images = []
    zero_fill_width = max(1, math.ceil(math.log10(args.num_samples + 1)))
    for idx in range(args.num_samples):
        image = generate_single_image(pipe, init_image, args)
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
        elapsed_time=generation_time,
        img2img_image_path=args.init_image_path,
        image_strength=args.strength)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from an image + a text prompt using SDXL Turbo.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--init-image-path", type=str, required=True, help="Path of image for img2img generation")
    parser.add_argument("--output-path", type=str, default="outputs/sdxl-turbo/img2img/", help="Output path for saving generated images (default: 'outputs/sdxl-turbo/img2img/')")
    # make sure that num_inference_steps * strength is larger or equal to 1
    parser.add_argument("--n-steps", type=int, default=2, help="Number of steps for image generation (default: 2)")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength for image generation, a float between 0.0 and 1.0 (default: 0.5)")
    # parser.add_argument("--high-noise-frac", type=float, default=0.8, help="High noise fraction for image generation (default: 0.8)")
    parser.add_argument("--log-path", type=str, default="logs/sdxl-turbo_img2img_log.txt", help="Output path for the log file (default: 'logs/sdxl-turbo_img2img_log.txt')")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate (default: 1)")

    args = parser.parse_args()
    main(args)
