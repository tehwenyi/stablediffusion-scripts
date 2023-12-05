import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datetime import datetime
import time

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

def _format_time(seconds):
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

def log(timestamp, args, log_filepath, n_images, output_folder, grid_output_path, elapsed_time, img2img_image_path=None, image_strength=None):
    """
    Log information to a file.

    Parameters:
        timestamp (str): The timestamp string.
        args: The command-line arguments.
        log_filepath (str): The path to the log file.
        n_images (int): The number of generated images.
        output_folder (Path): The folder where images are saved.
        grid_output_path (Path): The path to the grid image.
        generation_time (float): The time taken to generate the images.
        img2img_image_path (str, optional): The path to the image used for Img2Img processing.
    """
    generation_time = _format_time(elapsed_time)

    with open(log_filepath, 'a') as log_file:   
        log_file.write("==============================\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Prompt: {getattr(args, 'prompt', 'N/A')}\n")
        log_file.write("==============================\n")
        log_file.write(f"Negative Prompt: {getattr(args, 'negative_prompt', 'N/A')}\n")
        log_file.write(f"Number of Steps: {getattr(args, 'n_steps', 'N/A')}\n")
        log_file.write(f"High Noise Fraction: {getattr(args, 'high_noise_frac', 'N/A')}\n")
        log_file.write(f"Output Path: {getattr(args, 'output_path', 'N/A')}\n")
        log_file.write(f"Number of Generated Samples/Number of Intended Samples: {n_images}/{getattr(args, 'num_samples', 'N/A')}\n")
        log_file.write(f"Base Only: {getattr(args, 'base_only', 'N/A')}\n")
        log_file.write(f"Generation Time (HH:MM:SS): {generation_time}\n")
        if img2img_image_path: log_file.write(f"(Img2Img) Image Used: {img2img_image_path}\n")
        if image_strength: log_file.write(f"Image Strength: {image_strength}\n")
        log_file.write(f"Images Output Folder: {output_folder}\n")
        log_file.write(f"Images Grid Output Path: {grid_output_path}\n\n")


def _add_text_into_image(prompt_image, text, prompt_image_width):
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

    # Font size scaling with image size
    font_size = int(0.05 * prompt_image_width)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size) # one of the linux fonts

    # Wrap text
    wrapped_text = textwrap.fill(text, width=((prompt_image_width - 10) // font.getlength("A")))

    # Position of the text
    # Magic numbers, our favourite thing in the world
    x_position = 5
    y_position = 0

    prompt_draw.text((x_position, y_position), wrapped_text, fill=(0, 0, 0), font=font)

    return prompt_image

def save_image(image, idx, output_folder, timestamp, zero_fill_width):
    """
    Save an image to the specified folder with a formatted filename.

    Parameters:
        image (Image): The image to be saved.
        idx (int): The index of the image.
        output_folder (Path): The folder where the image will be saved.
        timestamp (str): The timestamp string.
        zero_fill_width (int): The width for zero-padding the index.
    """
    output_filename = timestamp + str(idx).zfill(zero_fill_width) + ".png"
    output_path = output_folder / output_filename
    image.save(output_path)

def save_images_grid(images, prompt, output_folder, timestamp):
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
    grid_output_filename = timestamp + "grid.png"
    output_path = output_folder / grid_output_filename
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

    prompt_image_width = grid_width
    prompt_image_height = cell_height
    prompt_image = Image.new("RGB", (prompt_image_width, prompt_image_height), (255, 255, 255))
    prompt_image = _add_text_into_image(prompt_image, prompt, prompt_image_width)

    # Calculate the row and column index for prompt_image
    grid_image.paste(prompt_image, (max(0, (grid_size[0] - 1) * cell_width), (grid_size[1] - 1) * cell_height))

    grid_image.save(output_path)
    print(f"Completed! Grid image saved at {output_path}")

    return str(output_path)

def load_image_from_path(image_path):
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')

    return img_rgb

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