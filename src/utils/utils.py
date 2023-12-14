import math
import textwrap
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

## Initialisation functions

def create_folder(output_path):
    """
    Create a folder at the specified path.

    Parameters:
        output_path (str): The path to the folder.

    Returns:
        Path: The Path object representing the created folder.
    """
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder

def create_timestamp():
    """
    Create a timestamp string in the format 'YYYYMMDD_HHMM_'.

    Returns:
        str: The timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M_")

def load_image_from_path(image_path):
    """
    Load an image from the specified file path and convert it to RGB format.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        Image: An RGB mode Image object representing the loaded image.
    """
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')

    return img_rgb

## Functions for saving images

def save_images(images, output_folder, timestamp, zero_fill_width):
    """
    Save images to the specified folder with formatted filenames.

    Parameters:
        images (list of Image): The images to be saved.
        output_folder (Path): The folder where the images will be saved.
        timestamp (str): The timestamp string.
        zero_fill_width (int): The width for zero-padding the index.
    """
    for idx, image in enumerate(images):
        output_filename = timestamp + str(idx).zfill(zero_fill_width) + ".png"
        output_path = output_folder / output_filename
        image.save(output_path)

    print(f"{idx + 1} images saved in {output_folder}")

def add_text_to_image(prompt_image, text, prompt_image_width):
    """
    Add wrapped text to an image.

    Parameters:
        prompt_image (Image): The image to which text will be added.
        text (str): The text to be added.
        prompt_image_width (int): The width of the image.

    Returns:
        Image: The modified image.
    """
    prompt_draw = ImageDraw.Draw(prompt_image)

    font_size = int(0.04 * prompt_image_width)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size) # one of the linux fonts

    wrapped_text = textwrap.fill(text, width=((prompt_image_width - 10) // font.getlength("A")))

    # Magic numbers, our favourite thing in the world
    x_position = 5
    y_position = 0

    prompt_draw.text((x_position, y_position), wrapped_text, fill=(0, 0, 0), font=font)

    return prompt_image

def save_images_and_prompt_grid(images, prompt, output_path):
    """
    Save a grid of images along with a prompt.

    Parameters:
        images (list): List of images to be included in the grid.
        prompt (str): The prompt text.
        output_folder (Path): The folder where the grid image will be saved.
        timestamp (str): The timestamp string.

    Returns:
        output_path (str): Output path of the grid image.
    """
    num_images = len(images) + 1
    grid_size = (int(math.sqrt(num_images)), math.ceil(num_images / int(math.sqrt(num_images))))
    cell_width, cell_height = images[0].size
    grid_width = grid_size[0] * cell_width
    grid_height = grid_size[1] * cell_height
    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for i, image in enumerate(images):
        row = i // grid_size[0]
        col = i % grid_size[0]
        grid_image.paste(image, (col * cell_width, row * cell_height))

    prompt_image_width = cell_width
    prompt_image_height = cell_height
    prompt_image = Image.new("RGB", (prompt_image_width, prompt_image_height), (255, 255, 255))
    prompt_image = add_text_to_image(prompt_image, prompt, prompt_image_width)

    prompt_row = max(0, (grid_size[0] - 1) * cell_width)
    prompt_col = (grid_size[1] - 1) * cell_height
    grid_image.paste(prompt_image, (prompt_row, prompt_col))

    grid_image.save(output_path)
    print(f"Completed! Grid image saved at {output_path}")

## Logging functions

def start_timer():
    """
    Start a timer and return the current time.

    Returns:
        float: The current time.
    """
    return time.time()

def stop_timer(start_time):
    """
    Stop the timer and return the elapsed time since the start.

    Parameters:
        start_time (float): The time when the timer was started.

    Returns:
        float: The elapsed time in seconds.
    """
    return time.time() - start_time

def format_elapsed_time(seconds):
    """
    Format the given time in seconds into HH:MM:SS.

    Parameters:
        seconds (float): The elapsed time in seconds.

    Returns:
        str: The formatted time string.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def log_generation_info(args, n_images, elapsed_time):
    """
    Log information to a file.

    Parameters:
        args: The command-line arguments.
        n_images (int): The number of generated images.
        elapsed_time (float): The time taken to generate the images.
    """
    create_folder(Path(args['log_path']).parent)
    generation_time = format_elapsed_time(elapsed_time)

    with open(args['log_path'], 'a') as log_file:   
        log_file.write("==============================\n")
        log_file.write(f"Timestamp: {args['timestamp']}\n")
        log_file.write(f"Prompt: {args.get('prompt', 'N/A')}\n\n")
        log_file.write(f"Negative Prompt: {args.get('negative_prompt', 'N/A')}\n")
        log_file.write(f"Number of Steps: {args.get('n_steps', 'N/A')}\n")
        log_file.write(f"High Noise Fraction: {args.get('high_noise_frac', 'N/A')}\n")
        log_file.write(f"Number of Generated Samples/Number of Intended Samples: {n_images}/{args.get('num_samples', 'N/A')}\n")
        log_file.write(f"(Only applicable for SDXL Txt2Img)Base Only: {args.get('base_only', 'N/A')}\n")
        log_file.write(f"(Only applicable for Img2Img) Image Used: {args.get('init_image_path', 'N/A')}\n")
        log_file.write(f"(Only applicable for Img2Img) Image Strength: {args.get('strength', 'N/A')}\n\n")
        log_file.write(f"Generation Time (HH:MM:SS): {generation_time}\n")
        log_file.write(f"Images Output Folder: {args.get('output_path', 'N/A')}\n")
        log_file.write(f"Images Grid Output Path: {args.get('grid_image_path', 'N/A')}\n")
        log_file.write("==============================\n\n")
