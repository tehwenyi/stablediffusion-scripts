import argparse

from models.SDXL_txt2img import SDXLTxt2Img
from models.SDXL_img2img import SDXLImg2Img
from models.SDXLturbo_txt2img import SDXLTurboTxt2Img
from models.SDXLturbo_img2img import SDXLTurboImg2Img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from an image + a text prompt using different models.")
    parser.add_argument("--model-name", type=str, required=True, choices=["sdxl-txt2img", "sdxl-img2img", "sdxlturbo-img2img", "sdxlturbo-txt2img"], help="Name of the model to use (sdxl-txt2img, sdxl-img2img, sdxlturbo-txt2img, sdxlturbo-img2img)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for image generation (default: '')")
    parser.add_argument("--init-image-path", type=str, default=None, help="Path of image for img2img generation")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength for image generation, a float between 0.0 and 1.0 (default: 0.5)")
    parser.add_argument("--high-noise-frac", type=float, default=0, help="High noise fraction for image generation (default: 0)")
    parser.add_argument("--n-steps", type=int, default=40, help="Number of steps for image generation (default: 40)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate (default: 1)")
    parser.add_argument("--output-path", type=str, default=None, help="Output path for saving generated images (default: 'output/[model-name]/')")
    parser.add_argument("--log-path", type=str, default=None, help="Output path for the log file (default: 'logs/[model-name]_log.txt')")
    parser.add_argument("--base-only", action="store_true", help="Only generate image through the base (default: False)")
    parser.add_argument("--load-lora-weights", type=str, default=None, help="Specify the path to the LORA weights for loading into the txt2img model (if any) (default: None)")

    args = parser.parse_args()

    output_path = args.output_path if args.output_path is not None else f"output/{args.model_name}/"
    log_path = args.log_path if args.log_path is not None else f"logs/{args.model_name}_log.txt"

    if args.model_name == "sdxl-txt2img":
        generator = SDXLTxt2Img(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            init_image_path=args.init_image_path,
            output_path=output_path,
            strength=args.strength,
            n_steps=args.n_steps,
            log_path=log_path,
            num_samples=args.num_samples,
            high_noise_frac=args.high_noise_frac,
            base_only=args.base_only,
            lora_weights=args.load_lora_weights
        )

    elif args.model_name == "sdxl-img2img":
        generator = SDXLImg2Img(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            init_image_path=args.init_image_path,
            output_path=output_path,
            strength=args.strength,
            n_steps=args.n_steps,
            high_noise_frac=args.high_noise_frac,
            log_path=log_path,
            num_samples=args.num_samples
        )

    elif args.model_name == "sdxlturbo-txt2img":
        generator = SDXLTurboTxt2Img(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            init_image_path=args.init_image_path,
            output_path=output_path,
            strength=args.strength,
            n_steps=args.n_steps,
            high_noise_frac=args.high_noise_frac,
            log_path=log_path,
            num_samples=args.num_samples,
            lora_weights=args.load_lora_weights
        )
    
    elif args.model_name == "sdxlturbo-img2img":
        generator = SDXLTurboImg2Img(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            init_image_path=args.init_image_path,
            output_path=output_path,
            strength=args.strength,
            n_steps=args.n_steps,
            high_noise_frac=args.high_noise_frac,
            log_path=log_path,
            num_samples=args.num_samples
        )

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    generator.main()
