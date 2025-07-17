import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import os # To check if the file exists

def convert_image_with_keras_utils(    
    image_path,
    target_size=(28, 28),
    color_mode="grayscale",
    normalize=True,
    invert_colors=True, # New parameter for inversion
    display_image=True
):
    """
    Converts a local image file to the specified target_size and color_mode using Keras utilities,
    reshapes it for model prediction, and optionally inverts its colors.

    Args:
        image_path (str): The **local path** to the input image file.
        target_size (tuple): The desired (width, height) of the output image.
        color_mode (str): "grayscale" or "rgb".
        normalize (bool): Whether to normalize pixel values to 0-1 range.
        invert_colors (bool): If True, inverts the pixel values (e.g., 0 becomes 1, 255 becomes 0).
        display_image (bool): Whether to display the processed image.

    Returns:
        numpy.ndarray: The processed image as a NumPy array with shape (1, target_height, target_width, channels),
                       normalized to 0-1, or None if an error occurs.
    """
    # Check if the file actually exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Please check the path.")
        return None

    try:
        # Step 1: Load the image using Keras utility
        img_pil = load_img(image_path, target_size=target_size, color_mode=color_mode)
        print(f"Loaded image via Keras utility. Size: {img_pil.size}, Mode: {img_pil.mode}")

        # Step 2: Convert the PIL Image to a NumPy array
        img_array = img_to_array(img_pil)
        print(f"Converted to NumPy array. Shape: {img_array.shape}, Dtype: {img_array.dtype}")

        # Step 3: Invert colors (if requested)
        if invert_colors:
            # Assuming 0-255 range for uint8, or 0-1 range for float (after initial normalization)
            if img_array.dtype == np.uint8:
                img_array = 255 - img_array # For 0-255 range
            elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = 1.0 - img_array # For 0-1 range
            else:
                print(f"Warning: Cannot invert colors for unknown dtype {img_array.dtype}. Skipping inversion.")
            print("Colors inverted.")


        # Step 4: Normalize pixel values (if required)
        # It's generally best to normalize *after* inversion if inversion works on the 0-255 range,
        # or consistently on the 0-1 range if inversion is done there.
        # Here, we normalize after potential 255-based inversion.
        if normalize:
            img_array = img_array.astype('float32') / 255.0
            print(f"Normalized pixel values to 0-1. Dtype: {img_array.dtype}")

        # Step 5: Add the batch dimension
        img_final = np.expand_dims(img_array, axis=0)
        print(f"Final shape for model input: {img_final.shape}")
        print(f"Final pixel value range: {img_final.min():.4f} to {img_final.max():.4f}")

        if display_image:
            plt.figure(figsize=(6, 6))
            title_text = f"Processed Local Image ({target_size[0]}x{target_size[1]} {color_mode}"
            if invert_colors:
                title_text += ", Inverted)"
            else:
                title_text += ")"
            plt.title(title_text)
            # Squeeze to remove batch and channel for display (imshow expects 2D or 3D without batch)
            plt.imshow(img_final.squeeze(), cmap='gray' if color_mode == 'grayscale' else None)
            plt.axis('off')
            plt.show()

        return img_final

    except Exception as e:
        print(f"An unexpected error occurred during image processing: {e}")
        return None