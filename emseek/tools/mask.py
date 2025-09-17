# import os
# import csv
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Check if scikit-image is available for connected-component analysis
# try:
#     from skimage import measure
#     SKIMAGE_AVAILABLE = True
# except ImportError:
#     SKIMAGE_AVAILABLE = False

# class MaskToNearestNeighborTool:
#     def __init__(self):
#         self.name = "MaskToNearestNeighborTool"
#         self.description = (
#             "MaskToNearestNeighborTool:\n"
#             "  - Generates a layered plot of nearest-neighbor distances (NND) and a Rayleigh distribution curve.\n"
#             "  - Plots measured (open circles) and random (solid circles) NNDs.\n"
#             "  - Includes a pink dashed line for an approximate bond distance (e.g., Pt−Pt).\n"
#             "  - Draws vertical dashed lines for the average measured and random distances."
#         )
    
#     def execute(self, input_json: dict, **kwargs) -> dict:
#         try:
#             # Only mask mode is supported: requires mask_image_path only.
#             mask_image_path = input_json.get("mask_image_path")
#             if not mask_image_path:
#                 raise ValueError("The 'mask_image_path' parameter is required!")
            
#             # Set default parameters (overridable via input JSON)
#             pixel_size_nm     = float(input_json.get("pixel_size_nm", 1.0))
#             chart_title       = input_json.get("title", "Rayleigh distribution of NNDs")
#             reference_line_nm = float(input_json.get("reference_line_nm", 1.0))
#             rayleigh_sigma_nm = float(input_json.get("rayleigh_sigma_nm", 0.5))
#             output_chart_path = input_json.get("output_chart_path", "nnd_plot.png")
            
#             # Verify that scikit-image is available
#             if not SKIMAGE_AVAILABLE:
#                 raise ImportError("The scikit-image library is not installed, which is required for connected-component analysis!")
            
#             # Open the mask image; convert it to grayscale for analysis
#             if not os.path.exists(mask_image_path):
#                 raise ValueError(f"Image file not found: {mask_image_path}")
#             mask_image = Image.open(mask_image_path).convert("L")
#             mask_array = np.array(mask_image)
            
#             # Identify connected components using scikit-image
#             nonzero_mask = (mask_array != 0)
#             labeled_components = measure.label(nonzero_mask, connectivity=1)
#             regions = measure.regionprops(labeled_components)
            
#             # Extract each connected component's centroid as an atom coordinate (x, y)
#             atom_coords = []
#             for region in regions:
#                 cy, cx = region.centroid  # centroid is given as (row, col)
#                 atom_coords.append((cx, cy))
            
#             if len(atom_coords) < 2:
#                 # Not enough atoms found; save an empty figure.
#                 plt.figure()
#                 plt.text(0.5, 0.5, 'Insufficient atoms: no NND computed.', horizontalalignment='center', verticalalignment='center')
#                 plt.axis('off')
#                 plt.savefig(output_chart_path, dpi=150)
#                 plt.close()
#                 return {
#                     "status": "success",
#                     "nnd_values_measured": [],
#                     "nnd_values_random": [],
#                     "output_chart_path": output_chart_path,
#                     "message": "Fewer than 2 atoms found; no NND computed. Empty figure saved."
#                 }
            
#             # Convert atom coordinates to a NumPy array
#             coords_np = np.array(atom_coords)
#             n_atoms = len(coords_np)
            
#             # 1) Compute measured nearest-neighbor distances
#             nnd_measured = self._compute_nnd(coords_np)
            
#             # 2) Generate random atom coordinates and compute their NND
#             x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
#             y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
#             random_coords = [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) for _ in range(n_atoms)]
#             random_coords_np = np.array(random_coords)
#             nnd_random = self._compute_nnd(random_coords_np)
            
#             # Convert distances to nanometers
#             nnd_measured_nm = nnd_measured * pixel_size_nm
#             nnd_random_nm   = nnd_random * pixel_size_nm
            
#             # Compute average nearest-neighbor distances
#             avg_measured = np.mean(nnd_measured_nm)
#             avg_random   = np.mean(nnd_random_nm)
            
#             # 3) Prepare the layered plot with Rayleigh distribution
#             fig, ax = plt.subplots(figsize=(6, 8))
            
#             # For aesthetic purposes, sort the measured NND values
#             sort_idx = np.argsort(nnd_measured_nm)
            
#             # Define the Rayleigh probability density function (PDF)
#             def rayleigh_pdf(r, sigma):
#                 return (r / (sigma**2)) * np.exp(-r**2 / (2 * sigma**2))
            
#             # Determine the x-axis limits based on the maximum NND and reference line value
#             x_max_val = max(3.0, nnd_measured_nm.max(), nnd_random_nm.max(), reference_line_nm) * 1.2
#             x_vals = np.linspace(0, x_max_val, 300)
            
#             # Y-axis spacing for each atom/layer
#             spacing = 1.0
#             for i, idx in enumerate(sort_idx):
#                 y_offset = i * spacing
#                 pdf_vals = rayleigh_pdf(x_vals, rayleigh_sigma_nm)
#                 pdf_scale = 1.0 / pdf_vals.max()  # Scale so the maximum value is ~1
#                 pdf_vals_scaled = pdf_vals * pdf_scale
#                 y_upper = y_offset + pdf_vals_scaled
#                 # Plot the filled area for the Rayleigh PDF
#                 ax.fill_between(x_vals, y_offset, y_upper, color='skyblue', alpha=0.5,
#                                 label='Rayleigh' if i == 0 else "")
#                 # Plot the measured NND (open circle)
#                 ax.scatter(nnd_measured_nm[idx], y_offset, facecolors='white', edgecolors='black', s=50,
#                            label='r_measured' if i == 0 else "")
#                 # Plot the random NND (solid circle)
#                 ax.scatter(nnd_random_nm[idx], y_offset, color='black', s=30,
#                            label='r_random' if i == 0 else "")
            
#             # Add a pink dashed reference line for the approximate bond distance
#             ax.axvline(reference_line_nm, color='magenta', linestyle='--', label='Approx. bond dist.')
#             # Draw vertical dashed lines for the average measured and random values
#             ax.axvline(avg_measured, color='black', linestyle='--', label='avg. r_measured')
#             ax.axvline(avg_random, color='gray', linestyle='--', label='avg. r_random')
            
#             ax.set_xlabel("Nearest Neighbor Distance (nm)")
#             ax.set_ylabel("Layer Number")
#             ax.set_title(chart_title)
#             ax.legend(loc='upper right', frameon=True)
#             ax.set_ylim(-spacing, n_atoms * spacing)
#             ax.set_xlim(0, x_max_val)
#             ax.grid(True, linestyle='--', alpha=0.5)
#             plt.tight_layout()
#             plt.savefig(output_chart_path, dpi=150)
#             plt.close()
            
#             return {
#                 "status": "success",
#                 "nnd_values_measured": nnd_measured_nm.tolist(),
#                 "nnd_values_random": nnd_random_nm.tolist(),
#                 "output_chart_path": output_chart_path,
#                 "message": "Rayleigh NND figure generated and saved successfully."
#             }
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "message": str(e)
#             }
    
#     def _compute_nnd(self, coords_np: np.ndarray) -> np.ndarray:
#         """
#         Compute the nearest-neighbor distance for each point in the provided coordinates.
#         Parameters:
#           coords_np - A two-dimensional NumPy array of shape (N, 2) with point coordinates.
#         Returns:
#           A NumPy array of nearest-neighbor distances (in pixel units).
#         """
#         n_points = len(coords_np)
#         distances = []
#         for i in range(n_points):
#             # Compute distance differences between point i and all points
#             dx = coords_np[:, 0] - coords_np[i, 0]
#             dy = coords_np[:, 1] - coords_np[i, 1]
#             dist_sq = dx**2 + dy**2
#             # Avoid zero distance for the point itself by assigning a large value
#             dist_sq[i] = 1e12  
#             min_dist = np.sqrt(dist_sq.min())
#             distances.append(min_dist)
#         return np.array(distances)

# # Example usage:
# if __name__ == "__main__":
#     tool = MaskToNearestNeighborTool()
#     # Replace 'path/to/your/mask_image.png' with the actual path to your mask image
#     input_data = {
#         "mask_image_path": "/Users/cgy/Code/MSAgent/history/9f450513-e7d3-48d2-b88c-728ce77e535b/mask.png",
#         # Optional: you can override defaults with:
#         # "pixel_size_nm": 0.8,
#         # "title": "NND Rayleigh Distribution",
#         # "reference_line_nm": 1.2,
#         # "rayleigh_sigma_nm": 0.6,
#         # "output_chart_path": "output_chart.png"
#     }
#     result = tool.execute(input_data)
#     print(result)


# import os
# import csv
# import math
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image
# from scipy.stats import norm

# # Check if scikit-image is available for connected component analysis
# try:
#     from skimage import measure
#     SKIMAGE_AVAILABLE = True
# except ImportError:
#     SKIMAGE_AVAILABLE = False

# class BaseTool:
#     """
#     BaseTool: A simple base class used for demonstration.
#     In practice, this can be extended as needed.
#     """
#     def __init__(self, name="", description=""):
#         self.name = name
#         self.description = description

#     def execute(self, input_json: dict, **kwargs) -> dict:
#         raise NotImplementedError("Subclasses must implement the execute method")

# class MaskOrPointsToAtomDensityTool(BaseTool):
#     def __init__(self):
#         description = """
#         MaskOrPointsToAtomDensityTool:
        
#         This tool computes the atom density (#atoms/area) from a given segmentation mask.
#         Currently, only "mask" mode is supported, and the input only requires the image file path.
#         All other parameters are set to default values.
        
#         If pixel_size_nm > 0 then the area is converted to nm^2; otherwise, it remains in pix^2.
        
#         The tool generates a scatter plot of "Area vs. Atom count" that includes diagonal lines 
#         representing constant density values, along with an inset showing the density distribution.
#         """
#         # """
#         # Input JSON format:
#         # {
#         #     "mask_image_path": "<string: path to the segmentation mask>",
#         #     "raw_image_path": "<string: optional, used to get the image width/height if provided>",
#         #     "output_chart_path": "<string: path to save the generated figure>"
#         # }
        
#         # Output JSON format (success):
#         # {
#         #     "status": "success",
#         #     "mask_result": {
#         #         "mask_image_path": "<string>",
#         #         "atom_count": <int>,
#         #         "area_value": <float>,
#         #         "density": <float>
#         #     },
#         #     "output_chart_path": "<string: path of the saved figure>",
#         #     "message": "Atom density chart generated and saved successfully."
#         # }
        
#         # On error:
#         # {
#         #     "status": "error",
#         #     "message": "<string: error message>"
#         # }
#         # """
#         super().__init__(name="MaskOrPointsToAtomDensityTool", description=description)
    
#     def compute_area_and_density(self, width_pix, height_pix, atom_count, pixel_size_nm):
#         """
#         Computes the area and atom density given the image dimensions and atom count.
#         If pixel_size_nm > 0, converts pixel dimensions to nm.
#         """
#         if width_pix <= 0 or height_pix <= 0:
#             return 0.0, 0.0
#         if pixel_size_nm > 0:
#             width_nm = width_pix * pixel_size_nm
#             height_nm = height_pix * pixel_size_nm
#             area_val = width_nm * height_nm
#         else:
#             area_val = width_pix * height_pix
#         density_val = atom_count / area_val if area_val > 0 else 0.0
#         return area_val, density_val

#     def execute(self, input_json: dict, **kwargs) -> dict:
#         try:
#             # Only input required is the image path; all other parameters use default settings.
#             mask_image_path = input_json.get("mask_image_path")
#             output_chart_path = os.path.join(input_json.get("output_chart_path"), 'atom_density_chart.png')
#             raw_image_path = input_json.get("raw_image_path", None)
            
#             # Default parameter settings
#             pixel_size_nm = 0.01                     # Defaults to pix^2 if not provided
#             chart_title = "Atom Density Chart"      # Chart title
#             density_lines = [0.02, 0.04, 0.06, 0.08, 0.1] # Default constant density lines to display
#             samples_label = "Sample"                # Default label for the data sample

#             # Check for required parameters.
#             if not mask_image_path or not output_chart_path:
#                 raise ValueError("Both 'mask_image_path' and 'output_chart_path' must be provided.")
#             if not SKIMAGE_AVAILABLE:
#                 raise ImportError("The scikit-image library is required for connected component analysis but is not available.")

#             # Retrieve image dimensions (prefer raw_image_path if provided)
#             width_pix, height_pix = 0, 0
#             if raw_image_path and os.path.exists(raw_image_path):
#                 with Image.open(raw_image_path) as raw_img:
#                     width_pix, height_pix = raw_img.size

#             # Open the mask image
#             if not os.path.exists(mask_image_path):
#                 raise ValueError(f"Mask image path does not exist: {mask_image_path}")
#             with Image.open(mask_image_path) as mask_img:
#                 mask_array = np.array(mask_img)
#                 if width_pix <= 0 or height_pix <= 0:
#                     width_pix, height_pix = mask_img.size

#             # Perform connected component analysis to count atoms (each non-zero region counts as one atom)
#             nonzero_mask = (mask_array != 0)
#             labeled = measure.label(nonzero_mask, connectivity=1)
#             regions = measure.regionprops(labeled)
#             atom_count = len(regions)
#             area_val, density_val = self.compute_area_and_density(width_pix, height_pix, atom_count, pixel_size_nm)

#             # Prepare the result dictionary for mask mode
#             mask_result = {
#                 "mask_image_path": mask_image_path,
#                 "atom_count": atom_count,
#                 "area_value": area_val,
#                 "density": density_val
#             }

#             # Prepare data for the scatter plot (only one sample in this case)
#             area_list = [area_val]
#             count_list = [atom_count]
#             density_list = [density_val]

#             # Create the scatter plot figure
#             fig, ax = plt.subplots(figsize=(7, 6))
#             ax.set_title(chart_title)
#             xlabel_str = "Area (nm^2)" if pixel_size_nm > 0 else "Area (pix^2)"
#             ax.set_xlabel(xlabel_str)
#             ax.set_ylabel("Atom count (-)")

#             # Plot the sample data point
#             ax.scatter(area_val, atom_count, color="blue", s=80, label=samples_label)

#             # Plot diagonal lines representing constant density lines.
#             max_area = max(area_list) if area_list else 1.0
#             x_vals = np.linspace(0.1 * max_area, 1.5 * max_area, 100)
#             for d_val in density_lines:
#                 y_vals = d_val * x_vals
#                 ax.plot(x_vals, y_vals, linestyle="--", color="gray", alpha=0.5)
#                 mid_idx = len(x_vals) // 2
#                 unit_str = "nm^-2" if pixel_size_nm > 0 else "pix^-2"
#                 ax.text(
#                     x_vals[mid_idx], y_vals[mid_idx],
#                     f"{d_val} {unit_str}",
#                     rotation=math.degrees(math.atan2(y_vals[-1] - y_vals[0], x_vals[-1] - x_vals[0])),
#                     color="gray", fontsize=8, alpha=0.7
#                 )

#             ax.legend(loc="upper left")
#             ax.grid(True, linestyle="--", alpha=0.5)

#             # Create an inset for the density distribution
#             inset_ax = fig.add_axes([0.55, 0.55, 0.35, 0.35])
#             inset_ax.set_title("Density distribution", fontsize=9)
#             # If more than one data point exists, display a histogram with a normal fit.
#             if len(density_list) > 1:
#                 mu, sigma = norm.fit(density_list)
#                 n, bins, _ = inset_ax.hist(density_list, bins='auto', color='skyblue',
#                                              alpha=0.7, edgecolor='black')
#                 x_fit = np.linspace(min(bins), max(bins), 100)
#                 pdf_fit = norm.pdf(x_fit, mu, sigma) * len(density_list) * (bins[1] - bins[0])
#                 inset_ax.plot(x_fit, pdf_fit, 'r--', linewidth=1.5, label=f"Normal fit\nμ={mu:.2e},σ={sigma:.2e}")
#                 inset_ax.legend(fontsize=7)
#             else:
#                 # For only one data point, mark it with a vertical line.
#                 inset_ax.axvline(density_list[0], color='red', linestyle='--')
#                 inset_ax.text(density_list[0], 0.5, f"{density_list[0]:.4f}", rotation=90, color='red')
#                 inset_ax.set_xlim(0, density_list[0] * 1.5 if density_list[0] > 0 else 1)
#                 inset_ax.set_ylim(0, 1)

#             xlabel_density = "Density (#/nm^2)" if pixel_size_nm > 0 else "Density (#/pix^2)"
#             inset_ax.set_xlabel(xlabel_density, fontsize=8)
#             inset_ax.set_ylabel("Count", fontsize=8)
#             inset_ax.tick_params(axis='both', which='major', labelsize=7)

#             plt.tight_layout()
#             plt.savefig(output_chart_path, dpi=150)
#             plt.close()

#             return {
#                 "status": "success",
#                 "mask_result": mask_result,
#                 "output_chart_path": output_chart_path,
#                 "message": "Atom density chart generated and saved successfully."
#             }

#         except Exception as e:
#             return {
#                 "status": "error",
#                 "message": str(e)
#             }

# # Example usage
# if __name__ == "__main__":
#     # Example input: only the mask image path and output chart path are required.
#     input_data = {
#         "mask_image_path": "/Users/cgy/Code/MSAgent/history/7019e1c5-67a6-42b3-aebf-8a7cf6ce7ed5/mask.png",
#         "output_chart_path": "./",
#         # raw_image_path is optional if you want to use a reference image to determine dimensions
#     }
    
#     tool = MaskOrPointsToAtomDensityTool()
#     result = tool.execute(input_data)
#     print(result)


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde
from skimage import measure

# Flag to indicate whether scikit-image is available (set to True by default)
SKIMAGE_AVAILABLE = True

class BaseTool:
    """
    BaseTool: A base class for all tools, providing a basic interface.
    """
    def __init__(self, name, description=""):
        self.name = name
        self.description = description

    def execute(self, input_json: dict, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement the execute method!")

class MaskToShapeDescriptorTool(BaseTool):
    """
    MaskToShapeDescriptorTool (KDE style):
    
    This tool computes shape descriptors (e.g., area, eccentricity, circularity, ellipticity)
    for each connected component in a segmentation mask image. It then generates a figure
    consisting of:
      - A 2D kernel density estimate (KDE) contour plot (default: area vs. eccentricity).
      - Marginal 1D KDE curves on the top and right sides.
    
    Input JSON format (only mask_image_path is required; all other parameters use defaults):
    {
        "mask_image_path": "<string, path to the mask image>"
    }
    
    Output JSON format (on success):
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
    
    Definitions:
      - area: region area in pix^2 or converted to nm^2 if pixel_size_nm > 0.
      - eccentricity: region properties eccentricity from skimage (0 = circle, 1 ~ line).
      - circularity (2D): 4*pi*area / perimeter^2.
      - ellipticity: 1 - (minor_axis_length / major_axis_length).
    """
    def __init__(self):
        description = (
            "MaskToShapeDescriptorTool (KDE style): computes shape descriptors and plots "
            "a 2D kernel density estimate with marginal 1D KDE curves."
        )
        super().__init__(name="MaskToShapeDescriptorTool", description=description)

    def execute(self, input_json: dict, **kwargs) -> dict:
        try:
            # 1. Parse input; require only mask_image_path, other parameters are default.
            mask_image_path = input_json.get("mask_image_path")
            if not mask_image_path:
                raise ValueError("The parameter 'mask_image_path' is required.")

            # Default output file path
            output_chart_path = os.path.join(input_json.get("output_chart_path", "./"), 'shape_descriptor_chart.png')
            # Default pixel size conversion (set to 0 to keep area as pix^2)
            pixel_size_nm = float(input_json.get("pixel_size_nm", 0.0))
            # Default descriptors
            descriptor_x = input_json.get("descriptor_x", "area")
            descriptor_y = input_json.get("descriptor_y", "eccentricity")
            # Default chart title
            chart_title = input_json.get("title", "Nanoparticle Shape Distribution")

            # 2. Verify that skimage is available
            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image module required for connected-component analysis is not installed.")

            # 3. Read the mask image and generate a non-zero mask
            mask_image = Image.open(mask_image_path).convert("L")
            mask_array = np.array(mask_image)
            nonzero_mask = (mask_array != 0)

            # 4. Perform connected-component analysis (4-connectivity)
            labeled = measure.label(nonzero_mask, connectivity=1)
            regions = measure.regionprops(labeled)

            # 5. Compute shape descriptors for each connected component
            descriptor_list = []
            for r in regions:
                # Area conversion if pixel_size_nm > 0
                area_pix = r.area
                area_val = area_pix * (pixel_size_nm ** 2) if pixel_size_nm > 0 else float(area_pix)
                # Eccentricity
                ecc_val = float(r.eccentricity)
                # Perimeter (avoid divide-by-zero)
                perimeter = r.perimeter if r.perimeter > 0 else 1e-6
                # Circularity
                circ_val = float(4.0 * np.pi * r.area / (perimeter ** 2))
                # Ellipticity calculation; avoid division by zero
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
            # If no particles are found, generate an empty figure with a notification.
            if particle_count == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.text(0.5, 0.5, "No particles found", ha="center", va="center", fontsize=12)
                plt.savefig(output_chart_path, dpi=150)
                plt.close()
                return {
                    "status": "success",
                    "particle_count": 0,
                    "descriptors": [],
                    "output_chart_path": output_chart_path,
                    "message": "No particles found. Empty figure saved."
                }

            # 6. Extract x and y data according to chosen descriptors
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

            # 7. Create the figure with main axes plus top/right marginal axes
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(4, 4)

            ax_main = fig.add_subplot(gs[1:4, 0:3])
            ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

            ax_main.set_title(chart_title)

            # 8. Create 2D KDE plot if there are enough data points
            if len(x_array) > 1:
                xy_data = np.vstack([x_array, y_array])
                kde_2d = gaussian_kde(xy_data)
                # Define grid with a slight extension beyond data range
                x_min, x_max = x_array.min(), x_array.max()
                y_min, y_max = y_array.min(), y_array.max()
                x_range = (x_max - x_min) * 0.05
                y_range = (y_max - y_min) * 0.05
                x_lin = np.linspace(x_min - x_range, x_max + x_range, 200)
                y_lin = np.linspace(y_min - y_range, y_max + y_range, 200)
                X, Y = np.meshgrid(x_lin, y_lin)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde_2d(positions), X.shape)
                levels = 10  # number of contour levels
                ax_main.contourf(X, Y, Z, levels=levels, cmap="autumn", alpha=0.4)
            else:
                # Only one data point available; plot scatter only.
                ax_main.scatter(x_array, y_array, color="orange")

            # Scatter the actual data points on the main axis.
            ax_main.scatter(x_array, y_array, color="orange", alpha=0.6, s=20)

            # Set the x-axis label; adjust units if descriptor_x is "area"
            if descriptor_x == "area":
                label_x = "Area (nm^2)" if pixel_size_nm > 0 else "Area (pix^2)"
            else:
                label_x = descriptor_x.capitalize()
            ax_main.set_xlabel(label_x)
            ax_main.set_ylabel(descriptor_y.capitalize())
            ax_main.grid(True, linestyle='--', alpha=0.5)

            # 9. Top marginal axis: plot the 1D KDE for x data.
            if len(x_array) > 1:
                kde_x = gaussian_kde(x_array)
                x_grid = np.linspace(x_array.min(), x_array.max(), 200)
                pdf_x = kde_x(x_grid)
                ax_top.fill_between(x_grid, pdf_x, color="orange", alpha=0.5)
            else:
                ax_top.axvline(x_array[0], color="orange", linestyle='--')
            ax_top.set_ylabel("Density")
            ax_top.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # 10. Right marginal axis: plot the 1D KDE for y data.
            if len(y_array) > 1:
                kde_y = gaussian_kde(y_array)
                y_grid = np.linspace(y_array.min(), y_array.max(), 200)
                pdf_y = kde_y(y_grid)
                ax_right.fill_betweenx(y_grid, pdf_y, color="orange", alpha=0.5)
            else:
                ax_right.axhline(y_array[0], color="orange", linestyle='--')
            ax_right.set_xlabel("Density")
            ax_right.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Adjust layout and save the figure.
            plt.tight_layout()
            plt.savefig(output_chart_path, dpi=150)
            plt.close()

            return {
                "status": "success",
                "particle_count": particle_count,
                "descriptors": descriptor_list,
                "output_chart_path": output_chart_path,
                "message": "KDE-based shape descriptor figure generated and saved successfully."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

# Example usage: only runs when the script is executed directly.
if __name__ == "__main__":
    import json
    # Only mask_image_path is required; the rest uses default settings.
    input_params = {"mask_image_path": "/Users/cgy/Code/MSAgent/history/7019e1c5-67a6-42b3-aebf-8a7cf6ce7ed5/mask.png"}
    tool = MaskToShapeDescriptorTool()
    result = tool.execute(input_params)
    print(json.dumps(result, indent=4))
