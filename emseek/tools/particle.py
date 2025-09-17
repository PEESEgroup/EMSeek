import os
import json
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gaussian_kde
from scipy.stats import norm
from PIL import Image, ImageDraw, ImageFont
from .base import BaseTool

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

class MaskToParticleSizeDistributionTool(BaseTool):
    """
    MaskToParticleSizeDistributionTool:
    
    This tool computes the particle diameter distribution from a segmentation mask.
    Each connected component in the mask is considered one particle. The diameter
    is taken from the region's 'equivalent_diameter' (or another measure if desired).
    
    Input JSON format:
    {
        "mask_image_path": "<string, path to the mask image>",
        "output_chart_path": "<string, path to save the particle size histogram>",
        "pixel_size_nm": "<float, optional, default=0.0; if >0 => convert from pix to nm>",
        "bins": "<int, optional, number of histogram bins (default=30)>",
        "title": "<string, optional, default='Particle Size Distribution'>",
        "x_label": "<string, optional, default='Particle Diameter (nm or pix)'>"
    }
    
    Output JSON format (success):
    {
        "status": "success",
        "particle_count": <int, total number of particles>,
        "diameters": [<float>, <float>, ...],  // list of diameters in nm or pix
        "mean_diameter": <float>,
        "std_diameter": <float>,
        "output_chart_path": "<string, path to the saved histogram>",
        "message": "Particle size distribution histogram generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string, error description>"
    }
    
    Explanation:
      - If pixel_size_nm > 0, diameters are reported in nm. Otherwise, in pixel units.
      - 'equivalent_diameter' from skimage is used, which is the diameter of a circle
        having the same area as the region.
      - The histogram is saved as a bar plot with the specified number of bins.
    """
    def __init__(self):
        description = (
            "MaskToParticleSizeDistributionTool: computes the particle diameter distribution "
            "from a segmentation mask. Each connected component is one particle, and the tool "
            "uses 'equivalent_diameter' for the particle size."
        )
        super().__init__(name="MaskToParticleSizeDistributionTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # 1. Parse input
            mask_image_path = input_json.get("mask_image_path")
            output_chart_path = input_json.get("output_chart_path")
            pixel_size_nm = float(input_json.get("pixel_size_nm", 0.0))
            bins = int(input_json.get("bins", 30))
            chart_title = input_json.get("title", "Particle Size Distribution")
            x_label = input_json.get("x_label", "Particle Diameter")

            if not mask_image_path or not output_chart_path:
                raise ValueError("mask_image_path and output_chart_path are required.")

            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image is required for connected-component analysis but not installed.")

            # 2. Read the mask image
            mask_image = Image.open(mask_image_path)
            mask_array = np.array(mask_image)

            # 3. Connected-component analysis
            from skimage import measure
            nonzero_mask = (mask_array != 0)
            labeled = measure.label(nonzero_mask, connectivity=1)
            regions = measure.regionprops(labeled)

            # 4. Extract particle diameters
            diameters = []
            for r in regions:
                # 'equivalent_diameter' is the diameter of a circle with the same area as the region
                d_pix = r.equivalent_diameter
                if pixel_size_nm > 0:
                    # convert from pix to nm
                    d_nm = d_pix * pixel_size_nm
                    diameters.append(d_nm)
                else:
                    # keep in pixel units
                    diameters.append(d_pix)

            particle_count = len(diameters)
            if particle_count == 0:
                return {
                    "status": "success",
                    "particle_count": 0,
                    "diameters": [],
                    "mean_diameter": 0.0,
                    "std_diameter": 0.0,
                    "output_chart_path": output_chart_path,
                    "message": "No particles found in the mask. Empty histogram saved."
                }

            diam_array = np.array(diameters)
            mean_diameter = float(np.mean(diam_array))
            std_diameter = float(np.std(diam_array))

            # 5. Plot histogram
            plt.figure(figsize=(7, 5))
            plt.hist(diam_array, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(chart_title)
            if pixel_size_nm > 0:
                plt.xlabel(f"{x_label} (nm)")
            else:
                plt.xlabel(f"{x_label} (pix)")
            plt.ylabel("Counts")
            # Optionally annotate mean & std
            plt.text(0.95, 0.95,
                     f"Mean: {mean_diameter:.2f}\nStd: {std_diameter:.2f}",
                     ha='right', va='top', transform=plt.gca().transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))

            plt.tight_layout()
            plt.savefig(output_chart_path, dpi=150)
            plt.close()

            return {
                "status": "success",
                "particle_count": particle_count,
                "diameters": diam_array.tolist(),
                "mean_diameter": mean_diameter,
                "std_diameter": std_diameter,
                "output_chart_path": output_chart_path,
                "message": "Particle size distribution histogram generated and saved successfully."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

class MaskToShapeDescriptorTool(BaseTool):
    """
    MaskToShapeDescriptorTool (KDE style):
    
    This tool computes shape descriptors (e.g., area, eccentricity, circularity, ellipticity)
    from each connected component in a segmentation mask. Then it creates a figure with:
      - A 2D kernel density estimate (KDE) contour plot (Area vs. Eccentricity by default).
      - Marginal 1D KDE curves on the top and right edges, matching the style of your second reference figure.
    
    Input JSON format:
    {
        "mask_image_path": "<string, path to the mask image>",
        "output_chart_path": "<string, path to save the shape descriptor figure>",
        "pixel_size_nm": "<float, optional, default=0.0; if >0 => convert from pix to nm^2 for area>",
        "descriptor_x": "<string, default='area'; e.g. 'area', 'circularity', 'ellipticity'>",
        "descriptor_y": "<string, default='eccentricity'; e.g. 'eccentricity', 'circularity'>",
        "title": "<string, optional, default='Nanoparticle Shape Distribution'>"
    }
    
    Output JSON format (success):
    {
        "status": "success",
        "particle_count": <int>,
        "descriptors": [
          {
            "area": <float>,
            "eccentricity": <float>,
            "circularity": <float>,
            "ellipticity": <float>
          },
          ...
        ],
        "output_chart_path": "<string>",
        "message": "KDE-based shape descriptor figure generated and saved successfully."
    }
    
    On error:
    {
        "status": "error",
        "message": "<string>"
    }
    
    Definitions:
      - area: region area in pix^2 or nm^2 if pixel_size_nm>0
      - eccentricity: skimage regionprops eccentricity (0 = circle, 1 = line)
      - circularity (2D): 4*pi*area / perimeter^2
      - ellipticity: (1 - minor_axis_length / major_axis_length)
    """

    def __init__(self):
        description = (
            "MaskToShapeDescriptorTool (KDE style): computes shape descriptors and plots "
            "a 2D kernel density estimate with marginal 1D KDE curves, matching the style of your second figure."
        )
        super().__init__(name="MaskToShapeDescriptorTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # 1. Parse input
            mask_image_path = input_json.get("mask_image_path")
            output_chart_path = input_json.get("output_chart_path")
            pixel_size_nm = float(input_json.get("pixel_size_nm", 0.0))
            descriptor_x = input_json.get("descriptor_x", "area")
            descriptor_y = input_json.get("descriptor_y", "eccentricity")
            chart_title = input_json.get("title", "Nanoparticle Shape Distribution")

            if not mask_image_path or not output_chart_path:
                raise ValueError("mask_image_path and output_chart_path are required.")

            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image is required for connected-component analysis but not installed.")

            # 2. Read mask image
            from skimage import measure
            mask_image = Image.open(mask_image_path)
            mask_array = np.array(mask_image)
            nonzero_mask = (mask_array != 0)

            # 3. Connected-component analysis
            labeled = measure.label(nonzero_mask, connectivity=1)
            regions = measure.regionprops(labeled)

            # 4. Compute shape descriptors
            descriptor_list = []
            for r in regions:
                # area
                area_pix = r.area
                if pixel_size_nm > 0:
                    area_val = area_pix * (pixel_size_nm**2)
                else:
                    area_val = float(area_pix)

                # eccentricity
                ecc_val = float(r.eccentricity)

                # perimeter
                perimeter = r.perimeter if r.perimeter > 0 else 1e-6
                # circularity
                circ_val = float(4.0 * np.pi * r.area / (perimeter**2))

                # ellipticity
                major_len = r.major_axis_length if r.major_axis_length > 0 else 1e-6
                minor_len = r.minor_axis_length
                ell_val = float(1.0 - (minor_len / major_len))

                descriptor_list.append({
                    "area": area_val,
                    "eccentricity": ecc_val,
                    "circularity": circ_val,
                    "ellipticity": ell_val
                })

            particle_count = len(descriptor_list)
            if particle_count == 0:
                return {
                    "status": "success",
                    "particle_count": 0,
                    "descriptors": [],
                    "output_chart_path": output_chart_path,
                    "message": "No particles found. Empty figure saved."
                }

            # 5. Extract x_data, y_data
            x_data = []
            y_data = []
            for d in descriptor_list:
                if descriptor_x not in d:
                    raise ValueError(f"Unsupported descriptor_x: {descriptor_x}")
                if descriptor_y not in d:
                    raise ValueError(f"Unsupported descriptor_y: {descriptor_y}")
                x_data.append(d[descriptor_x])
                y_data.append(d[descriptor_y])

            x_array = np.array(x_data, dtype=float)
            y_array = np.array(y_data, dtype=float)

            # 6. Create figure with main axes + top/right marginal axes
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(4, 4)

            ax_main = fig.add_subplot(gs[1:4, 0:3])
            ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

            ax_main.set_title(chart_title)

            # 7. 2D KDE on (x_array, y_array)
            #    We'll create a grid and evaluate gaussian_kde
            if len(x_array) > 1:
                xy_data = np.vstack([x_array, y_array])
                kde_2d = gaussian_kde(xy_data)
                # define grid
                x_min, x_max = x_array.min(), x_array.max()
                y_min, y_max = y_array.min(), y_array.max()
                x_range = (x_max - x_min) * 0.05
                y_range = (y_max - y_min) * 0.05
                x_lin = np.linspace(x_min - x_range, x_max + x_range, 200)
                y_lin = np.linspace(y_min - y_range, y_max + y_range, 200)
                X, Y = np.meshgrid(x_lin, y_lin)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde_2d(positions), X.shape)

                # contourf with alpha
                levels = 10  # number of contour levels
                ax_main.contourf(X, Y, Z, levels=levels, cmap="autumn", alpha=0.4)
            else:
                # if there's only 1 data point, no meaningful KDE
                x_min, x_max = min(x_array), max(x_array)
                y_min, y_max = min(y_array), max(y_array)
                # just plot a single point or skip
                ax_main.scatter(x_array, y_array, color="orange")

            # 8. A faint scatter of actual points (optional)
            ax_main.scatter(x_array, y_array, color="orange", edgecolor="none", alpha=0.6, s=20)

            # Axes labels
            if descriptor_x == "area":
                label_x = "Area (nm^2)" if pixel_size_nm > 0 else "Area (pix^2)"
            else:
                label_x = descriptor_x.capitalize()
            ax_main.set_xlabel(label_x)

            ax_main.set_ylabel(descriptor_y.capitalize())
            ax_main.grid(True, linestyle='--', alpha=0.5)

            # 9. Top axis: 1D KDE for x
            if len(x_array) > 1:
                kde_x = gaussian_kde(x_array)
                x_grid = np.linspace(x_min, x_max, 200)
                pdf_x = kde_x(x_grid)
                ax_top.fill_between(x_grid, pdf_x, color="orange", alpha=0.5)
            else:
                # single data => just draw a line or skip
                ax_top.axvline(x_array[0], color='orange', linestyle='--')
            ax_top.set_ylabel("Density")
            ax_top.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # 10. Right axis: 1D KDE for y (horizontal)
            if len(y_array) > 1:
                kde_y = gaussian_kde(y_array)
                y_grid = np.linspace(y_min, y_max, 200)
                pdf_y = kde_y(y_grid)
                ax_right.fill_betweenx(y_grid, pdf_y, color="orange", alpha=0.5)
            else:
                ax_right.axhline(y_array[0], color='orange', linestyle='--')
            ax_right.set_xlabel("Density")
            ax_right.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(output_chart_path, dpi=150)
            plt.close()

            return {
                "status": "success",
                "particle_count": len(descriptor_list),
                "descriptors": descriptor_list,
                "output_chart_path": output_chart_path,
                "message": "KDE-based shape descriptor figure generated and saved successfully."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }