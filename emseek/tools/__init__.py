from .base import BaseTool
from .math_tools import SquareRootTool, ExponentiationTool, ArithmeticTool
from .image import (
    ImageMergeTool,
    ImageBrightnessHistogramTool,
    AutoScaleAtomicSizeCountTool,
    PointToDefectSegmentationTool,
    PointToAtomSegmentationTool,
    MaskToNearestNeighborTool,
    MaskOrPointsToAtomDensityTool,
)
from .particle import MaskToParticleSizeDistributionTool, MaskToShapeDescriptorTool

# Unified functional wrappers (no heavy deps required for import)
from .analysis_tools import (
    hyperspy_pca_denoise,
    py4dstem_fft_calibration,
    atomap_peak_detect,
    skimage_edges,
    pymatgen_cif_summary,
    materials_project_query,
)

# Registry of available tools for programmatic dispatch
TOOL_REGISTRY = {
    # Analysis functions
    'hyperspy_pca': hyperspy_pca_denoise,
    'py4dstem_calibration': py4dstem_fft_calibration,
    'atomap_peaks': atomap_peak_detect,
    'skimage_edges': skimage_edges,
    'pymatgen_summary': pymatgen_cif_summary,
    'materials_project': materials_project_query,

    # Class-based tools (wrap with execute)
    'image_merge': ImageMergeTool().execute,
    'image_histogram': ImageBrightnessHistogramTool().execute,
    'auto_atomic_size_count': AutoScaleAtomicSizeCountTool().execute,
    'defect_seg_from_points': PointToDefectSegmentationTool().execute,
    'atom_seg_from_points': PointToAtomSegmentationTool().execute,
    'mask_nnd': MaskToNearestNeighborTool().execute,
    'atom_density': MaskOrPointsToAtomDensityTool().execute,
    'particle_size_distribution': MaskToParticleSizeDistributionTool().execute,
    'shape_descriptor_kde': MaskToShapeDescriptorTool().execute,
}

# Aliases for more natural names
TOOL_ALIASES = {
    'hyperspy': 'hyperspy_pca',
    'pca_denoise': 'hyperspy_pca',
    'py4dstem': 'py4dstem_calibration',
    'calibration': 'py4dstem_calibration',
    'atomap': 'atomap_peaks',
    'peaks': 'atomap_peaks',
    'edges': 'skimage_edges',
    'scikit-image': 'skimage_edges',
    'skimage': 'skimage_edges',
    'pymatgen': 'pymatgen_summary',
    'mp': 'materials_project',
}


def resolve_tool(name: str) -> str:
    key = (name or '').strip().lower()
    return TOOL_ALIASES.get(key, key)


def run_tool(tool_name: str, params: dict, workdir: str = None):
    name = resolve_tool(tool_name)
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    fn = TOOL_REGISTRY[name]
    # Simple adapter: function vs. class execute
    if fn in (hyperspy_pca_denoise, skimage_edges):
        # functions that require out_dir
        out_dir = params.get('out_dir') or (workdir or '.')
        image_key = 'image_path' if 'image_path' in params else params.get('input_image_path')
        if name == 'hyperspy_pca':
            return fn(params.get('image_path') or image_key, out_dir)
        if name == 'skimage_edges':
            return fn(params.get('image_path') or image_key, out_dir)
    if fn is materials_project_query:
        return fn(params.get('query', 'Si'), params.get('api_key'))
    if fn is pymatgen_cif_summary:
        return fn(params.get('cif_path'))
    if fn is py4dstem_fft_calibration:
        return fn(params.get('image_path'))
    if fn is atomap_peak_detect:
        return fn(params.get('image_path'))
    # Class-based execute signature expects a single dict
    return fn(params)


def list_tools(include_aliases: bool = False):
    """Return a dictionary of available tool names (and optionally aliases).

    - When include_aliases=False (default), returns canonical tool keys.
    - When include_aliases=True, returns a mapping {alias: canonical} merged with canonical keys.
    """
    tools = dict(TOOL_REGISTRY)
    if include_aliases:
        # Map aliases to canonical entries for discoverability
        tools = {**{k: TOOL_ALIASES[k] for k in TOOL_ALIASES}, **tools}
    return tools
