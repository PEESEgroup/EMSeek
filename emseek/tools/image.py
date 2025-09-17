import os
import json
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import norm
from PIL import Image, ImageDraw, ImageFont
from .base import BaseTool

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

class ImageMergeTool(BaseTool):
    """
    ImageMergeTool: Merges a segmentation mask with the original image.
    
    If the mask is white, randomly replace white pixels with a random color before blending.

    Input JSON format:
    {
        "original_image_path": "<string, path to original image>",
        "segmentation_mask_path": "<string, path to segmentation mask image>",
        "output_image_path": "<string, path where the merged image will be saved>",
        "alpha": "<optional float between 0 and 1, blending factor (default 0.5)>"
    }

    Output JSON format on success:
    {
        "status": "success",
        "output_image_path": "<string, path to merged image>",
        "message": "Image merged successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    """
    def __init__(self):
        description = (
            "ImageMergeTool: Merges a segmentation mask with the original image.\n"
            "If the mask is white, randomly replace white pixels with a random color before blending.\n\n"
            "Input JSON format:\n"
            "{\n"
            "    \"original_image_path\": \"<string, path to original image>\",\n"
            "    \"segmentation_mask_path\": \"<string, path to segmentation mask image>\",\n"
            "    \"output_image_path\": \"<string, path where the merged image will be saved>\",\n"
            "    \"alpha\": \"<optional float between 0 and 1, blending factor (default 0.5)>\"\n"
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            "    \"status\": \"success\",\n"
            "    \"output_image_path\": \"<string, path to merged image>\",\n"
            "    \"message\": \"Image merged successfully.\"\n"
            "}\n\n"
            "On error:\n"
            "{\n"
            "    \"status\": \"error\",\n"
            "    \"message\": \"<string, error description>\"\n"
            "}"
        )
        super().__init__(name="ImageMergeTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        """
        Execute the image merging operation.
        If the mask is white, randomly replace white pixels with a random color before blending.

        Parameters:
            input_json (dict): Input dictionary following the format specified in the tool's description.
            
        Returns:
            dict: Output dictionary in the format described in the tool's description.
        """
        try:
            original_image_path = input_json.get("original_image_path")
            segmentation_mask_path = input_json.get("segmentation_mask_path")
            output_image_path = input_json.get("output_image_path")
            alpha = float(input_json.get("alpha", 0.5))
            
            # Validate required fields.
            if not original_image_path or not segmentation_mask_path or not output_image_path:
                raise ValueError(
                    "Missing required input fields: original_image_path, "
                    "segmentation_mask_path, or output_image_path."
                )
            
            # Open the original image and segmentation mask.
            original = Image.open(original_image_path).convert("RGBA")
            mask = Image.open(segmentation_mask_path).convert("RGBA")
            
            # Resize the segmentation mask to match the original image, if necessary.
            if original.size != mask.size:
                mask = mask.resize(original.size)
            
            # Randomly choose a color for replacing white pixels.
            # We keep the alpha = 255 so it's fully opaque in those regions.
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255
            )
            
            # Replace white pixels (255,255,255,anything) with the random color
            mask_data = mask.load()
            width, height = mask.size
            for y in range(height):
                for x in range(width):
                    r, g, b, a = mask_data[x, y]
                    # If it's white in RGB channels (ignoring alpha),
                    # replace with the random color
                    if (r, g, b) == (255, 255, 255):
                        mask_data[x, y] = (random_color[0], random_color[1], random_color[2], a)
            
            # Blend the two images using the specified alpha value.
            merged = Image.blend(original, mask, alpha=alpha)
            
            # Save the merged image.
            merged.save(output_image_path)
            
            return {
                "status": "success",
                "output_image_path": output_image_path,
                "message": "Image merged successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


class ImageBrightnessHistogramTool(BaseTool):
    """
    ImageBrightnessHistogramTool: Computes the brightness histogram of an input image and saves a visually appealing histogram plot.

    Input JSON format:
    {
        "input_image_path": "<string, path to the input image>",
        "output_histogram_path": "<string, path where the histogram image will be saved>",
        "bins": "<optional integer, number of bins for the histogram (default: 256)>",
        "colormap": "<optional string, name of the colormap to use (default: 'viridis')>"
    }

    Output JSON format on success:
    {
        "status": "success",
        "output_histogram_path": "<string, path to the saved histogram image>",
        "message": "Brightness histogram generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    """
    def __init__(self):
        description = (
            "ImageBrightnessHistogramTool: Computes the brightness histogram of an input image and saves a visually appealing "
            "histogram plot as an image file.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "input_image_path": "<string, path to the input image>",\n'
            '    "output_histogram_path": "<string, path where the histogram image will be saved>",\n'
            '    "bins": "<optional integer, number of bins for the histogram (default: 256)>",\n'
            '    "colormap": "<optional string, name of the colormap to use (default: \'viridis\')>"\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "output_histogram_path": "<string, path to the saved histogram image>",\n'
            '    "message": "Brightness histogram generated and saved successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<string, error description>"\n'
            "}"
        )
        super().__init__(name="ImageBrightnessHistogramTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # Retrieve input parameters.
            input_image_path = input_json.get("input_image_path")
            output_histogram_path = input_json.get("output_histogram_path")
            bins = int(input_json.get("bins", 256))
            colormap = input_json.get("colormap", "viridis")
            
            if not input_image_path or not output_histogram_path:
                raise ValueError("Missing required fields: 'input_image_path' or 'output_histogram_path'.")
            
            # Open and convert the input image to grayscale.
            image = Image.open(input_image_path)
            image_gray = image.convert("L")
            
            # Convert image to numpy array and compute the histogram.
            np_image = np.array(image_gray)
            hist, bin_edges = np.histogram(np_image, bins=bins, range=(0, 255))
            
            # Create a plot for the brightness histogram.
            plt.figure(figsize=(10, 6))
            plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], 
                    color=plt.get_cmap(colormap)(0.6), edgecolor="black")
            plt.xlabel("Brightness")
            plt.ylabel("Frequency")
            plt.title("Brightness Histogram")
            plt.grid(True, linestyle="--", alpha=0.7)
            
            # Save the histogram plot as an image.
            plt.savefig(output_histogram_path, bbox_inches="tight")
            plt.close()
            
            return {
                "status": "success",
                "output_histogram_path": output_histogram_path,
                "message": "Brightness histogram generated and saved successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class AutoScaleAtomicSizeCountTool(BaseTool):
    """
    AutoScaleAtomicSizeCountTool:
      - Ignores pixels with value 0 (background).
      - Performs connected-component analysis on non-zero pixels.
      - Measures the size (pixel count) of each connected component (atom).
      - Automatically determines size bins using np.histogram(..., bins='auto').
      - Generates a bar chart showing how many atoms fall into each automatically determined size bin.
    
    Input JSON format:
    {
        "mask_image_path": "<string, path to the mask image>",
        "output_chart_path": "<string, path where the chart will be saved>",
        "title": "<optional string, title of the chart (default: 'Auto-Scaled Atomic Size Count')>",
        "colormap": "<optional string, name of the colormap (default: 'tab20')>"
    }
    
    Output JSON format on success:
    {
        "status": "success",
        "bin_edges": [<float>, ...],   // Automatically determined bin boundaries
        "bin_counts": [<int>, ...],     // Count of atoms in each bin
        "output_chart_path": "<string, path to the saved chart>",
        "message": "Atomic size count chart generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    """
    def __init__(self):
        description = (
            "AutoScaleAtomicSizeCountTool:\n"
            "  - Ignores pixels with value 0 (background).\n"
            "  - Performs connected-component analysis on non-zero pixels.\n"
            "  - Measures the size (pixel count) of each connected component (atom).\n"
            "  - Automatically determines size bins using np.histogram(..., bins='auto').\n"
            "  - Produces a bar chart of the number of atoms in each automatically determined size bin.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "mask_image_path": "<string>",\n'
            '    "output_chart_path": "<string>",\n'
            '    "title": "<optional string, default: \'Auto-Scaled Atomic Size Count\'>",\n'
            '    "colormap": "<optional string, default: \'tab20\'>"\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "bin_edges": [<float>, ...],\n'
            '    "bin_counts": [<int>, ...],\n'
            '    "output_chart_path": "<string>",\n'
            '    "message": "Atomic size count chart generated and saved successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<string>"\n'
            "}"
        )
        super().__init__(name="AutoScaleAtomicSizeCountTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # Extract input parameters
            mask_image_path = input_json.get("mask_image_path")
            output_chart_path = input_json.get("output_chart_path")
            chart_title = input_json.get("title", "Auto-Scaled Atomic Size Count")
            colormap = input_json.get("colormap", "tab20")

            # Check required fields
            if not mask_image_path or not output_chart_path:
                raise ValueError("Missing required fields: 'mask_image_path' or 'output_chart_path'.")

            # Ensure scikit-image is available
            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image is required for connected-component analysis but not installed.")

            # Load the mask image and convert to a NumPy array
            mask_image = Image.open(mask_image_path)
            mask_array = np.array(mask_image)

            # Create a boolean mask for non-zero pixels (ignoring background)
            nonzero_mask = (mask_array != 0)

            # Perform connected-component analysis on non-zero pixels using 4-connectivity
            labeled_components = measure.label(nonzero_mask, connectivity=1)
            regions = measure.regionprops(labeled_components)

            # Get the size (pixel count) for each connected component (atom)
            sizes = [r.area for r in regions]

            # If no atoms are found, return an empty result
            if len(sizes) == 0:
                return {
                    "status": "success",
                    "bin_edges": [],
                    "bin_counts": [],
                    "output_chart_path": output_chart_path,
                    "message": "No non-zero atoms found. Empty chart saved."
                }

            # Automatically determine bins using NumPy's histogram with 'auto'
            bin_counts, bin_edges = np.histogram(sizes, bins='auto')

            # Create a bar chart of the atomic size counts
            plt.figure(figsize=(10, 6))
            # Compute bin centers for the x-axis
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bar_width = bin_edges[1:] - bin_edges[:-1]

            # Generate colors for the bars using the specified colormap
            cmap = plt.get_cmap(colormap)
            colors = cmap(np.linspace(0.1, 0.9, len(bin_counts)))

            # Plot the bars
            plt.bar(bin_centers, bin_counts, width=bar_width, color=colors, align='center', edgecolor='black')
            plt.xlabel("Atom Size (pixel count)")
            plt.ylabel("Number of Atoms")
            plt.title(chart_title)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Save the chart to the specified output path
            plt.savefig(output_chart_path, bbox_inches="tight")
            plt.close()

            # Return success JSON with bin edges and counts
            return {
                "status": "success",
                "bin_edges": bin_edges.tolist(),
                "bin_counts": bin_counts.tolist(),
                "output_chart_path": output_chart_path,
                "message": "Atomic size count chart generated and saved successfully."
            }
        except Exception as e:
            # Return error JSON if an exception occurs
            return {
                "status": "error",
                "message": str(e)
            }

class PointToDefectSegmentationTool(BaseTool):
    """
    PointToDefectSegmentationTool:
      - Reads an input image.
      - Reads a text file containing point coordinates. Each line in the file is formatted as:
            <category> <x_coordinate> <y_coordinate>
        For example:
            1 100 34
            1 168 41
            2 209 81
      - Draws a hollow circle (with specified radius and line width) for each point.
      - Uses different colors for different defect categories.
      - Saves the output image with defects marked.
    
    Input JSON format:
    {
        "input_image_path": "<string, path to the input image>",
        "points_file_path": "<string, path to the text file containing point coordinates>",
        "output_image_path": "<string, path where the defect segmentation image will be saved>",
        "circle_radius": "<optional integer, radius of the circle (default: 10)>",
        "line_width": "<optional integer, line width of the circle outline (default: 2)>"
    }
    
    Output JSON format on success:
    {
        "status": "success",
        "output_image_path": "<string, path to the saved defect segmentation image>",
        "message": "Defect segmentation image generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    """
    def __init__(self):
        description = (
            "PointToDefectSegmentationTool:\n"
            "  - Reads an input image and a text file with defect point coordinates.\n"
            "  - The text file should have lines formatted as: <category> <x_coordinate> <y_coordinate>.\n"
            "  - Draws hollow circles for each defect on the input image. Different defect categories are marked with different colors.\n"
            "  - Saves the resulting defect segmentation image.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "input_image_path": "<string, path to the input image>",\n'
            '    "points_file_path": "<string, path to the text file with coordinates>",\n'
            '    "output_image_path": "<string, path where the defect segmentation image will be saved>",\n'
            '    "circle_radius": "<optional integer, default: 10>",\n'
            '    "line_width": "<optional integer, default: 2>"\n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "output_image_path": "<string, path to the saved defect segmentation image>",\n'
            '    "message": "Defect segmentation image generated and saved successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<string, error description>"\n'
            "}"
        )
        super().__init__(name="PointToDefectSegmentationTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # Extract input parameters.
            input_image_path = input_json.get("input_image_path")
            points_file_path = input_json.get("points_file_path")
            output_image_path = input_json.get("output_image_path")
            circle_radius = int(input_json.get("circle_radius", 10))
            line_width = int(input_json.get("line_width", 2))
            
            # Validate required fields.
            if not input_image_path or not points_file_path or not output_image_path:
                raise ValueError("Missing required fields: input_image_path, points_file_path, or output_image_path.")
            
            # Open the input image.
            image = Image.open(input_image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            # Define a mapping from category to color.
            # You can add more colors if more categories are expected.
            category_colors = {
                "1": "red",
                "2": "green",
                "3": "blue",
                "4": "orange",
                "5": "purple"
            }
            default_color = "yellow"  # Fallback color if category is not found.
            
            # Read the point coordinates from the text file.
            with open(points_file_path, "r") as file:
                lines = file.readlines()
            
            # Process each line in the points file.
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Each line should have three parts: category, x, and y.
                parts = line.split()
                if len(parts) < 3:
                    continue
                category, x_str, y_str = parts[0], parts[1], parts[2]
                try:
                    x = float(x_str)
                    y = float(y_str)
                except ValueError:
                    continue
                
                # Get the color for this category.
                color = category_colors.get(category, default_color)
                
                # Calculate the bounding box for the hollow circle.
                left = x - circle_radius
                top = y - circle_radius
                right = x + circle_radius
                bottom = y + circle_radius
                
                # Draw a hollow circle (only the outline).
                draw.ellipse([(left, top), (right, bottom)], outline=color, width=line_width)
            
            # Save the output image.
            image.save(output_image_path)
            
            return {
                "status": "success",
                "output_image_path": output_image_path,
                "message": "Defect segmentation image generated and saved successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class PointToAtomSegmentationTool(BaseTool):
    """
    PointToAtomSegmentationTool:
      - Reads an input image.
      - Reads a CSV file containing atom point coordinates.
        The CSV file should have a header (e.g., ",X,Y") and rows formatted as:
            <atom_id>,<X coordinate>,<Y coordinate>
        For example:
            1,152,351
            2,174,332
            3,247,266
            ...
      - Draws a hollow circle at each atom location using the same color for all atoms.
      - Saves the resulting atom segmentation image.
    
    Input JSON format:
    {
        "input_image_path": "<string, path to the input image>",
        "points_csv_path": "<string, path to the CSV file with coordinates>",
        "output_image_path": "<string, path where the atom segmentation image will be saved>",
        "circle_radius": "<optional integer, radius of the circle (default: 10)>",
        "line_width": "<optional integer, line width of the circle outline (default: 2)>",
        "color": "<optional string, color to use for all atoms (default: 'red')>"
    }
    
    Output JSON format on success:
    {
        "status": "success",
        "output_image_path": "<string, path to the saved atom segmentation image>",
        "message": "Atom segmentation image generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    """
    def __init__(self):
        description = (
            "PointToAtomSegmentationTool:\n"
            "  - Reads an input image and a CSV file with atom point coordinates.\n"
            "  - The CSV file should have a header (e.g., ',X,Y') and each subsequent row formatted as:\n"
            "        <atom_id>,<X coordinate>,<Y coordinate>\n"
            "  - Draws hollow circles at each atom location using the same color for all atoms.\n"
            "  - Saves the resulting atom segmentation image.\n\n"
            "Input JSON format:\n"
            "{\n"
            '    "input_image_path": "<string, path to the input image>",\n'
            '    "points_csv_path": "<string, path to the CSV file with coordinates>",\n'
            '    "output_image_path": "<string, path where the atom segmentation image will be saved>",\n'
            '    "circle_radius": "<optional integer, default: 10>",\n'
            '    "line_width": "<optional integer, default: 2>",\n'
            '    "color": "<optional string, default: \'red\'>" \n'
            "}\n\n"
            "Output JSON format on success:\n"
            "{\n"
            '    "status": "success",\n'
            '    "output_image_path": "<string, path to the saved atom segmentation image>",\n'
            '    "message": "Atom segmentation image generated and saved successfully."\n'
            "}\n\n"
            "On error:\n"
            "{\n"
            '    "status": "error",\n'
            '    "message": "<string, error description>"\n'
            "}"
        )
        super().__init__(name="PointToAtomSegmentationTool", description=description)
    
    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # Extract input parameters
            input_image_path = input_json.get("input_image_path")
            points_csv_path = input_json.get("points_csv_path")
            output_image_path = input_json.get("output_image_path")
            circle_radius = int(input_json.get("circle_radius", 10))
            line_width = int(input_json.get("line_width", 2))
            color = input_json.get("color", "red")
            
            # Validate required fields
            if not input_image_path or not points_csv_path or not output_image_path:
                raise ValueError("Missing required fields: input_image_path, points_csv_path, or output_image_path.")
            
            # Open the input image and prepare to draw
            image = Image.open(input_image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            # Read the CSV file and process each row
            with open(points_csv_path, newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                header = next(csvreader, None)  # Skip header row if present
                for row in csvreader:
                    # Each row should have three values: atom_id, X, and Y.
                    if len(row) < 3:
                        continue
                    try:
                        x = float(row[1].strip())
                        y = float(row[2].strip())
                    except ValueError:
                        continue
                    
                    # Calculate the bounding box for the hollow circle
                    left = x - circle_radius
                    top = y - circle_radius
                    right = x + circle_radius
                    bottom = y + circle_radius
                    
                    # Draw a hollow circle (outline only)
                    draw.ellipse([(left, top), (right, bottom)], outline=color, width=line_width)
            
            # Save the output image
            image.save(output_image_path)
            
            return {
                "status": "success",
                "output_image_path": output_image_path,
                "message": "Atom segmentation image generated and saved successfully."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class MaskToNearestNeighborTool(BaseTool):
    def __init__(self):
        description = (
            "MaskToNearestNeighborTool:\n"
            "  - Produces a layered plot of nearest-neighbor distances (NND) with Rayleigh distribution.\n"
            "  - Plots measured (open circles) vs. random (solid circles) NNDs.\n"
            "  - Includes a pink dashed line for an approximate bond distance (e.g. Pt−Pt).\n"
            "  - Draws vertical dashed lines for the average measured and random distances.\n"
            "Input JSON and Output JSON are described in the class docstring."
        )
        super().__init__(name="MaskToNearestNeighborTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            mode = input_json.get("mode")
            mask_image_path = input_json.get("mask_image_path")
            points_csv_path = input_json.get("points_csv_path")
            output_chart_path = input_json.get("output_chart_path")
            pixel_size_nm = float(input_json.get("pixel_size_nm", 1.0))
            chart_title = input_json.get("title", "Rayleigh distribution of NNDs")
            reference_line_nm = float(input_json.get("reference_line_nm", 1.0))
            rayleigh_sigma_nm = float(input_json.get("rayleigh_sigma_nm", 0.5))

            # Validate required fields
            if not mode or not output_chart_path:
                raise ValueError("Missing required field: 'mode' or 'output_chart_path'.")

            # Collect atom coordinates
            atom_coords = []

            if mode == "mask":
                if not mask_image_path:
                    raise ValueError("mask_image_path is required when mode='mask'.")
                if not SKIMAGE_AVAILABLE:
                    raise ImportError("scikit-image is required for connected-component analysis but not installed.")

                mask_image = Image.open(mask_image_path)
                mask_array = np.array(mask_image)

                from skimage import measure
                nonzero_mask = (mask_array != 0)
                labeled_components = measure.label(nonzero_mask, connectivity=1)
                regions = measure.regionprops(labeled_components)

                for r in regions:
                    cy, cx = r.centroid  # (row, col) => (y, x)
                    atom_coords.append((cx, cy))

            elif mode == "points":
                if not points_csv_path:
                    raise ValueError("points_csv_path is required when mode='points'.")
                if not os.path.exists(points_csv_path):
                    raise ValueError(f"CSV file not found: {points_csv_path}")

                with open(points_csv_path, newline='') as csvfile:
                    csvreader = csv.reader(csvfile)
                    header = next(csvreader, None)
                    for row in csvreader:
                        if len(row) < 2:
                            continue
                        try:
                            x_val = float(row[-2])
                            y_val = float(row[-1])
                            atom_coords.append((x_val, y_val))
                        except ValueError:
                            continue
            else:
                raise ValueError("Invalid mode. Must be 'mask' or 'points'.")

            if len(atom_coords) < 2:
                return {
                    "status": "success",
                    "nnd_values_measured": [],
                    "nnd_values_random": [],
                    "output_chart_path": output_chart_path,
                    "message": "Fewer than 2 atoms found; no NND computed. Empty figure saved."
                }

            coords_np = np.array(atom_coords)
            n_atoms = len(coords_np)

            # --- 1) Measured NND ---
            nnd_measured = self._compute_nnd(coords_np)

            # --- 2) Random NND ---
            x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
            y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()

            random_coords = []
            for _ in range(n_atoms):
                rx = random.uniform(x_min, x_max)
                ry = random.uniform(y_min, y_max)
                random_coords.append((rx, ry))
            random_coords_np = np.array(random_coords)
            nnd_random = self._compute_nnd(random_coords_np)

            # Convert distances to nm
            nnd_measured_nm = nnd_measured * pixel_size_nm
            nnd_random_nm = nnd_random * pixel_size_nm

            # Averages
            avg_measured = np.mean(nnd_measured_nm)
            avg_random = np.mean(nnd_random_nm)

            # --- 3) Prepare layered plot ---
            fig, ax = plt.subplots(figsize=(6, 8))

            # Sort by measured NND for aesthetics
            sort_idx = np.argsort(nnd_measured_nm)

            # Rayleigh PDF
            def rayleigh_pdf(r, sigma):
                return (r / (sigma**2)) * np.exp(-r**2 / (2*sigma**2))

            # x-range for Rayleigh distribution
            x_max_val = max(3.0, nnd_measured_nm.max(), nnd_random_nm.max(), reference_line_nm) * 1.2
            x_vals = np.linspace(0, x_max_val, 300)

            # We'll space each atom by 1.0 on the y-axis
            spacing = 1.0

            for i, idx in enumerate(sort_idx):
                y_offset = i * spacing
                pdf_vals = rayleigh_pdf(x_vals, rayleigh_sigma_nm)
                pdf_scale = 1.0 / pdf_vals.max()  # scale so the max is ~1
                pdf_vals_scaled = pdf_vals * pdf_scale
                y_upper = y_offset + pdf_vals_scaled
                # Fill
                ax.fill_between(x_vals, y_offset, y_upper, color='skyblue', alpha=0.5, label='Rayleigh' if i==0 else "")

                # Plot measured (open circle)
                ax.scatter(nnd_measured_nm[idx], y_offset, facecolors='white', edgecolors='black', s=50,
                           label='r_measured' if i==0 else "")

                # Plot random (solid circle)
                ax.scatter(nnd_random_nm[idx], y_offset, color='black', s=30,
                           label='r_random' if i==0 else "")

            # Pink reference line
            ax.axvline(reference_line_nm, color='magenta', linestyle='--', label='Approx. bond dist.')

            # Average lines: black for measured, gray for random
            ax.axvline(avg_measured, color='black', linestyle='--', label='avg. r_measured')
            ax.axvline(avg_random, color='gray', linestyle='--', label='avg. r_random')

            ax.set_xlabel("NND (nm)")
            ax.set_ylabel("Image (-)")
            ax.set_title(chart_title)

            ax.legend(loc='upper right', frameon=True)
            ax.set_ylim(-spacing, n_atoms * spacing)
            ax.set_xlim(0, x_max_val)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(output_chart_path, dpi=150)
            plt.close()

            return {
                "status": "success",
                "nnd_values_measured": nnd_measured_nm.tolist(),
                "nnd_values_random": nnd_random_nm.tolist(),
                "output_chart_path": output_chart_path,
                "message": "Rayleigh NND figure generated and saved successfully."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _compute_nnd(self, coords_np):
        """
        Compute the nearest-neighbor distance for each point in coords_np.
        Returns a NumPy array of distances in pixel units.
        """
        n_points = len(coords_np)
        dists = []
        for i in range(n_points):
            dx = coords_np[:, 0] - coords_np[i, 0]
            dy = coords_np[:, 1] - coords_np[i, 1]
            dist_sq = dx**2 + dy**2
            dist_sq[i] = 1e12  # avoid zero distance to itself
            min_dist = np.sqrt(dist_sq.min())
            dists.append(min_dist)
        return np.array(dists)

class MaskOrPointsToAtomDensityTool(BaseTool):
    """
    MaskOrPointsToAtomDensityTool:
    
    This tool computes single-atom density (#atoms / area) from either:
      - A single segmentation mask (mode='mask'), or
      - Multiple CSV files of point coordinates (mode='points').
    
    If pixel_size_nm > 0, it converts pixel area to nm^2. Otherwise, it stays in pix^2.
    
    In 'points' mode, you can provide a list of CSV files. Each CSV file is treated as a separate sample.
    The tool computes (area, atom_count, density) for each sample and plots multiple data points in the same figure.
    
    Input JSON format:
    {
        "mode": "mask" or "points",
        
        // If mode='mask':
        "mask_image_path": "<string, path to the mask>",
        
        // If mode='points':
        "points_csv_paths": ["<string>", "<string>", ...], // a list of CSV files
        
        "raw_image_path": "<string, optional, if provided, used to get image width/height>",
        
        "output_chart_path": "<string, required, path to save the figure>",
        "pixel_size_nm": "<float, optional, default=0.0; if >0 => convert pix to nm>",
        "title": "<string, optional, default='Atom Density Chart'>",
        "density_lines": "[float, ...], optional, constant density lines to draw",
        "samples_label": "<string, optional, label prefix or fallback name>"
    }
    
    Output JSON format (success):
    {
        "status": "success",
        "results": [
            {
                "csv_path": "<string>",
                "atom_count": <int>,
                "area_value": <float>,
                "density": <float>
            },
            ...
        ],
        "mask_result": {
            "mask_image_path": "<string>",
            "atom_count": <int>,
            "area_value": <float>,
            "density": <float>
        } // if mode='mask'
        "output_chart_path": "<string, path to the saved figure>",
        "message": "Atom density chart generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error message>"
    }
    """
    def __init__(self):
        description = (
            "MaskOrPointsToAtomDensityTool: computes single-atom density (#/area) from a mask or multiple CSV files. "
            "If pixel_size_nm > 0, density is in #/nm^2; otherwise #/pix^2. "
            "Generates an 'Area vs. Atom count' scatter plot with diagonal lines for constant density, "
            "and an inset for density distribution across multiple samples."
        )
        super().__init__(name="MaskOrPointsToAtomDensityTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            mode = input_json.get("mode")
            mask_image_path = input_json.get("mask_image_path")
            points_csv_paths = input_json.get("points_csv_paths", [])  # list of CSV files
            raw_image_path = input_json.get("raw_image_path")  # optional
            output_chart_path = input_json.get("output_chart_path")
            pixel_size_nm = float(input_json.get("pixel_size_nm", 0.0))
            chart_title = input_json.get("title", "Atom Density Chart")
            density_lines = input_json.get("density_lines", [0.2, 0.4, 0.6, 0.8, 1.0])
            samples_label = input_json.get("samples_label", "Sample")

            if not mode or not output_chart_path:
                raise ValueError("Must provide 'mode' ('mask' or 'points') and 'output_chart_path'.")

            # We'll store results in a list or dict
            results = []
            mask_result = None

            # We will store data for the final scatter plot
            area_list = []
            count_list = []
            density_list = []

            # Helper function to compute area and density for a given (width_pix, height_pix, atom_count)
            def compute_area_and_density(width_pix, height_pix, atom_count):
                if width_pix <= 0 or height_pix <= 0:
                    return 0.0, 0.0
                if pixel_size_nm > 0:
                    width_nm = width_pix * pixel_size_nm
                    height_nm = height_pix * pixel_size_nm
                    area_val = width_nm * height_nm
                else:
                    area_val = width_pix * height_pix
                density_val = atom_count / area_val if area_val > 0 else 0.0
                return area_val, density_val

            # 1) If mode='mask', handle a single mask
            if mode == "mask":
                if not mask_image_path:
                    raise ValueError("mask_image_path is required if mode='mask'.")
                if not SKIMAGE_AVAILABLE:
                    raise ImportError("scikit-image is required for connected-component analysis but not installed.")

                # If raw_image_path is provided, open it
                width_pix, height_pix = 0, 0
                if raw_image_path and os.path.exists(raw_image_path):
                    with Image.open(raw_image_path) as raw_img:
                        width_pix, height_pix = raw_img.size

                # Open the mask
                mask_image = Image.open(mask_image_path)
                mask_array = np.array(mask_image)

                if width_pix <= 0 or height_pix <= 0:
                    # fallback to mask image size
                    width_pix, height_pix = mask_image.size

                # connected components
                from skimage import measure
                nonzero_mask = (mask_array != 0)
                labeled = measure.label(nonzero_mask, connectivity=1)
                regions = measure.regionprops(labeled)
                atom_count = len(regions)  # each region is an atom

                area_val, density_val = compute_area_and_density(width_pix, height_pix, atom_count)

                mask_result = {
                    "mask_image_path": mask_image_path,
                    "atom_count": atom_count,
                    "area_value": area_val,
                    "density": density_val
                }
                area_list.append(area_val)
                count_list.append(atom_count)
                density_list.append(density_val)

            # 2) If mode='points', handle multiple CSV files
            elif mode == "points":
                if not points_csv_paths or len(points_csv_paths) == 0:
                    raise ValueError("points_csv_paths must be a non-empty list if mode='points'.")

                for csv_path in points_csv_paths:
                    if not os.path.exists(csv_path):
                        continue
                    # If raw_image_path is provided, open it to get size
                    width_pix, height_pix = 0, 0
                    if raw_image_path and os.path.exists(raw_image_path):
                        with Image.open(raw_image_path) as raw_img:
                            width_pix, height_pix = raw_img.size

                    # read points from csv
                    atom_coords = []
                    with open(csv_path, newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader, None)
                        for row in reader:
                            if len(row) < 2:
                                continue
                            try:
                                x_val = float(row[-2])
                                y_val = float(row[-1])
                                atom_coords.append((x_val, y_val))
                            except:
                                pass

                    atom_count = len(atom_coords)
                    if (width_pix <= 0 or height_pix <= 0) and atom_count > 0:
                        xs = [c[0] for c in atom_coords]
                        ys = [c[1] for c in atom_coords]
                        width_pix = math.ceil(max(xs) - min(xs))
                        height_pix = math.ceil(max(ys) - min(ys))

                    area_val, density_val = compute_area_and_density(width_pix, height_pix, atom_count)
                    results.append({
                        "csv_path": csv_path,
                        "atom_count": atom_count,
                        "area_value": area_val,
                        "density": density_val
                    })
                    area_list.append(area_val)
                    count_list.append(atom_count)
                    density_list.append(density_val)
            else:
                raise ValueError("mode must be 'mask' or 'points'.")

            # If we have no data
            if len(area_list) == 0:
                return {
                    "status": "success",
                    "results": results,
                    "mask_result": mask_result,
                    "output_chart_path": output_chart_path,
                    "message": "No valid samples found. No figure generated."
                }

            # 3) Plot the figure
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_title(chart_title)
            if pixel_size_nm > 0:
                ax.set_xlabel("Area (nm^2)")
            else:
                ax.set_xlabel("Area (pix^2)")
            ax.set_ylabel("Atom count (-)")

            # Plot each sample
            for i, (area_val, count_val, dens_val) in enumerate(zip(area_list, count_list, density_list)):
                # Use a label for the first point or if you want to label each sample individually
                lbl = samples_label if i == 0 else None
                ax.scatter(area_val, count_val, color="blue", s=80, label=lbl)

            # Draw diagonal lines for constant density
            max_area = max(area_list) if len(area_list) > 0 else 1.0
            x_vals = np.linspace(0.1 * max_area, 1.5 * max_area, 100)
            for d_val in density_lines:
                y_vals = d_val * x_vals
                ax.plot(x_vals, y_vals, linestyle="--", color="gray", alpha=0.5)
                mid_idx = len(x_vals) // 2
                unit_str = "nm^-2" if pixel_size_nm > 0 else "pix^-2"
                ax.text(
                    x_vals[mid_idx], y_vals[mid_idx],
                    f"{d_val} {unit_str}",
                    rotation=math.degrees(math.atan2((y_vals[-1] - y_vals[0]), (x_vals[-1] - x_vals[0]))),
                    color="gray", fontsize=8, alpha=0.7
                )

            # If we have at least one label, show legend
            if samples_label:
                ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Inset: density distribution
            inset_ax = fig.add_axes([0.55, 0.55, 0.35, 0.35])
            inset_ax.set_title("Density distribution", fontsize=9)

            # If multiple data points, show a histogram + normal fit
            if len(density_list) > 1:
                mu, sigma = norm.fit(density_list)
                n, bins, patches = inset_ax.hist(density_list, bins='auto', color='skyblue',
                                                 alpha=0.7, edgecolor='black')
                x_fit = np.linspace(min(bins), max(bins), 100)
                pdf_fit = norm.pdf(x_fit, mu, sigma) * len(density_list) * (bins[1] - bins[0])
                inset_ax.plot(x_fit, pdf_fit, 'r--', linewidth=1.5,
                              label=f"Normal fit\nμ={mu:.2e},σ={sigma:.2e}")
                inset_ax.legend(fontsize=7)
            else:
                # Only one data point
                density_val = density_list[0]
                inset_ax.axvline(density_val, color='red', linestyle='--')
                inset_ax.text(density_val, 0.5, f"{density_val:.4f}", rotation=90, color='red')
                if density_val > 0:
                    inset_ax.set_xlim(0, density_val * 1.5)
                else:
                    inset_ax.set_xlim(0, 1)
                inset_ax.set_ylim(0, 1)

            if pixel_size_nm > 0:
                inset_ax.set_xlabel("Density (#/nm^2)", fontsize=8)
            else:
                inset_ax.set_xlabel("Density (#/pix^2)", fontsize=8)
            inset_ax.set_ylabel("Count", fontsize=8)
            inset_ax.tick_params(axis='both', which='major', labelsize=7)

            plt.tight_layout()
            plt.savefig(output_chart_path, dpi=150)
            plt.close()

            return {
                "status": "success",
                "results": results,
                "mask_result": mask_result,
                "output_chart_path": output_chart_path,
                "message": "Atom density chart generated and saved successfully."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }