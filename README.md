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

Requirement: `torch >= 2.0`

Tested on `Python 3.10`.

### Download Weights

Download the latest weights for the following components into the checkpoints folder:

- Latest weights for base are [here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main). As of 4 Dec 2023, the latest weights are `sd_xl_base_1.0.safetensors` (used by SDXL Turbo).
- Latest weights for refiner are [here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/tree/main). As of 4 Dec 2023, the latest weights are `sd_xl_refiner_1.0.safetensors` (optional, used by SDXL Text to Image).

### SDXL

#### Text to Image
- Requires text prompt only
- Consists of 2 components: [base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0). You may choose to use only the base or allow the base to generate an image which will then be run through the refiner for better results.

Example:
```bash
python3 sdxl_txt2img.py --prompt "A cinematic shot of a baby racoon wearing an intricate italian priest robe." --num-samples 5
```

#### Image to Image
- Requires both a text and an image prompt

Example:
```bash
python3 sdxl_img2img.py --prompt "A cinematic shot of a baby racoon wearing an intricate italian priest robe." --init-image-path "example-racoon.png" --num-samples 5
```

### SDXL Turbo

#### Text to Image
- Requires text prompt only

Example:
```bash
python3 sdxl-turbo_txt2img.py --prompt "A cinematic shot of a baby racoon wearing an intricate italian priest robe." --num-samples 5
```

Note (from [here](https://huggingface.co/stabilityai/sdxl-turbo)): SDXL-Turbo does not make use of `guidance_scale` or `negative_prompt`, we disable it with `guidance_scale=0.0`. Preferably, the model generates images of size `512x512` but higher image sizes work as well. A single step is enough to generate high quality images.

#### Image to Image
- Requires both a text and an image prompt

Example:
```bash
python3 sdxl-turbo_img2img.py --prompt "A cinematic shot of a baby racoon wearing an intricate italian priest robe." --init-image-path "example-racoon.png" --num-samples 5
```

Note (from [here](https://huggingface.co/stabilityai/sdxl-turbo)): When using SDXL-Turbo for image-to-image generation, make sure that `num_inference_steps` * `strength` is larger or equal to `1`. The image-to-image pipeline will run for `int(num_inference_steps * strength)` steps, e.g. `0.5 * 2.0 = 1` step in the default parameters.

For more information on available parameters, please refer to the respective scripts.

## To-Do List
- [ ] Add Dockerfile
