# Stable Diffusion Scripts

Collection of scripts to simplify the use of stable diffusion models, specifically SDXL and SDXL Turbo.

## Supported Models
- SDXL
- SDXL Turbo

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tehwenyi/stablediffusion-scripts.git

1. Navigate to the repository:

   ```bash
   cd stablediffusion-scripts
   ```

1. Set up and activate virtual environment:

   ```bash
   conda create --name stablediffusion python=3.10
   ```

   ```bash
   conda activate stablediffusion
   ```

1. Install required packages:

   ```bash
   pip install diffusers transformers accelerate --upgrade
   pip install xformers==0.0.22
   ```

## Usage

This repository contains Python code for generating images using various models. The supported models are:

- SDXL Text-to-Image (sdxl-txt2img)
- SDXL Image-to-Image (sdxl-img2img)
- SDXLturbo Text-to-Image (sdxlturbo-txt2img)
- SDXLturbo Image-to-Image (sdxlturbo-img2img)

Requirement: `torch >= 2.0`

Tested on `Python 3.10`

### Download Weights

Download the latest weights for the following components into the `checkpoints` folder:

- Latest weights for base are [here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main). As of 4 Dec 2023, the latest weights are `sd_xl_base_1.0.safetensors` (used by SDXL Turbo).
- Latest weights for refiner are [here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/tree/main). As of 4 Dec 2023, the latest weights are `sd_xl_refiner_1.0.safetensors` (optional, used by SDXL Text to Image).

### Command-line Arguments

The script `generate_images.py` takes the following command-line arguments:

`--model-name`: Name of the model to use (choose from: sdxl-txt2img, sdxl-img2img, sdxlturbo-txt2img, sdxlturbo-img2img).
`--prompt`: Prompt for image generation.
`--negative-prompt`: Negative prompt for image generation (default: '').
`--init-image-path`: Path of the image for img2img generation.
`--strength`: Strength for image generation, a float between 0.0 and 1.0 (default: 0.5).
`--high-noise-frac`: High noise fraction for image generation (default: 0).
`--n-steps`: Number of steps for image generation (default: 40).
`--num-samples`: Number of samples to generate (default: 1).
`--output-path`: Output path for saving generated images (default: 'output/[model-name]/').
`--log-path`: Output path for the log file (default: 'logs/[model-name]_log.txt').
`--use-fp16`: Load weights from a specified variant filename 'fp16' (default: False).
`--local-weights-path`: Local file path(s) for the model weights (optional, please indicate both paths if using base and refiner for SDXL_txt2img).
`--lora-weights`: Specify the path to the LORA weights for loading into the txt2img model (if any) (default: None).
`--base-only`: Only generate an image through the base (default: False).

### Examples

#### Generate Image using SDXL Text-to-Image

```bash
python generate_images.py --model-name sdxl-txt2img --prompt "Your text prompt here"
```

#### Generate Image using SDXL Image-to-Image

```bash
python generate_images.py --model-name sdxl-img2img --prompt "Your text prompt here" --init-image-path "/path/to/your/image.png"
```

Alternatively, refer to the bash script `run_generate_images.sh`.

## To-Do List
- [x] Add Dockerfile
- [ ] Add lora information into log file
- [ ] Allow loading model weights from local folder - Incomplete, only done with SDXL_txt2img
