import numpy as np
from skimage.morphology import remove_small_objects, ball, closing

def apply_3d_morphological_closing(mask_3d: np.ndarray, min_sizes: dict, radii: dict) -> np.ndarray:
    """
    Applies 3D post-processing to a multi-class segmentation mask.
    Removes small connected components and applies 3D morphological closing 
    to preserve anatomical coherence of the tumor volumes.

    Parameters
    ----------
    mask_3d : np.ndarray
        3D array containing predicted labels (0=background, 1=GTVp, 2=GTVn).
    min_sizes : dict
        Minimum object size (in voxels) for each class (e.g., {1: 50, 2: 1200}).
    radii : dict
        Radius of the spherical structuring element for morphological closing (e.g., {1: 1, 2: 3}).

    Returns
    -------
    np.ndarray
        Refined 3D mask as a uint8 array.
    """
    refined_mask = np.zeros_like(mask_3d, dtype=np.uint8)

    for label_val in [1, 2]:
        # Extract binary mask for the current clinical class
        bin_mask = (mask_3d == label_val)

        # Remove spurious small isolated artifacts
        bin_mask = remove_small_objects(bin_mask, min_size=min_sizes.get(label_val, 20))

        # Apply 3D morphological closing to fill internal gaps
        strel = ball(radii.get(label_val, 1))
        closed_mask = closing(bin_mask, strel)

        # Assign refined regions back to the output volume
        refined_mask[closed_mask] = label_val

    return refined_mask
