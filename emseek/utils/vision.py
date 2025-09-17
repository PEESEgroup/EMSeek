import os
import cv2
import random
import numpy as np
from PIL import Image

def random_bright_color():
    """
    Generate a random bright color (in BGR format) to be used for drawing segmentation contours.
    """
    hue = random.randint(0, 179)
    sat = random.randint(200, 255)
    val = random.randint(200, 255)
    hsv_color = np.uint8([[[hue, sat, val]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
    return bgr_color

def overlay_mask_on_image(image, mask, alpha=0.5, extra_radius=5):
    """
    Overlay the segmentation mask onto the original image and draw the contours.
    
    Parameters:
      - image: The original image (PIL.Image or numpy array)
      - mask: The segmentation mask (numpy array)
      - alpha: Transparency coefficient
      - extra_radius: Additional radius for the contour circle
    Returns:
      - The image with the overlay (numpy array)
    """

    # Convert the image to a numpy array
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output = image.copy()

    # Convert mask type and adjust mask size to match the image
    mask = mask.astype(np.uint8)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Randomly generate bright colors for filling and drawing contours (requires the random_bright_color function)
    fill_color = random_bright_color()
    contour_color = random_bright_color()
    
    for cnt in contours:
        mask_contour = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_contour, [cnt], -1, 255, thickness=cv2.FILLED)
        color_mask = np.zeros_like(image)
        color_mask[:] = fill_color
        output = np.where(mask_contour[:, :, np.newaxis] == 255,
                          cv2.addWeighted(output, 1 - alpha, color_mask, alpha, 0),
                          output)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius) + extra_radius
        cv2.circle(output, center, radius, contour_color, thickness=2)
    return output

def convert_yolo_to_bbox(bbox_data, image_size):
    """
    Convert YOLO format bounding box data to image pixel coordinates.
    bbox_data should include x_center, y_center, width, and height (all relative proportions).
    """
    x_center = float(bbox_data['x_center'])
    y_center = float(bbox_data['y_center'])
    width = float(bbox_data['width'])
    height = float(bbox_data['height'])
    img_width, img_height = image_size

    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = width * img_width
    h_px = height * img_height

    top_left = (int(x_center_px - w_px / 2), int(y_center_px - h_px / 2))
    bottom_right = (int(x_center_px + w_px / 2), int(y_center_px + h_px / 2))
    return top_left, bottom_right

def crop_patch(image, bbox_coords):
    """
    Crop the image region based on the bounding box coordinates.
    bbox_coords: (top_left, bottom_right)
    """
    top_left, bottom_right = bbox_coords
    return image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

def convert_to_pil(arr):
    """
    Convert a numpy array to a PIL Image object (if needed).
    """
    if isinstance(arr, np.ndarray):
        return Image.fromarray(arr)
    elif isinstance(arr, Image.Image):
        return arr
    else:
        raise ValueError("Input is neither a numpy array nor a PIL Image object.")

def save_image_file(image, folder, prefix):
    """
    Save the image to the specified folder with the filename format {prefix}.png.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{prefix}.png"
    filepath = os.path.join(folder, filename)
    image = convert_to_pil(image)
    if image.mode == 'F':
        image = image.convert("RGB")
    image.save(filepath)
    print(f"Saved {prefix} image at:", filepath)
    return filepath