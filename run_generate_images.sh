#!/bin/bash

# Example for SDXL Img2Img

MODEL_NAME="sdxl-img2img" # ["sdxl-txt2img", "sdxl-img2img", "sdxlturbo-img2img", "sdxlturbo-txt2img"]
PROMPT="Your prompt text"
NEGATIVE_PROMPT="Your negative prompt here"
INIT_IMAGE_PATH="path/to/your/image.jpg"
STRENGTH=0.35
HIGH_NOISE_FRAC=0
N_STEPS=40
NUM_SAMPLES=4

python generate_images.py \
    --model-name "$MODEL_NAME" \
    --prompt "$PROMPT" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --init-image-path "$INIT_IMAGE_PATH" \
    --strength $STRENGTH \
    --high-noise-frac $HIGH_NOISE_FRAC \
    --n-steps $N_STEPS \
    --num-samples $NUM_SAMPLES
