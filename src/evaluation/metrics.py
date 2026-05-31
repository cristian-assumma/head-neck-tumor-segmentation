import numpy as np

def compute_dice_coefficient(mask_manual: np.ndarray, mask_auto: np.ndarray, label: int) -> float:
    """
    Computes the Dice Similarity Coefficient (DSC) between a ground truth mask and an automated prediction.

    Parameters
    ----------
    mask_manual : np.ndarray
        Ground truth segmentation mask.
    mask_auto : np.ndarray
        Predicted segmentation mask.
    label : int
        The specific class label to evaluate.

    Returns
    -------
    float
        The computed DSC value. Returns np.nan if the label is entirely absent in both masks.
    """
    manual_label_mask = (mask_manual == label)
    auto_label_mask = (mask_auto == label)

    intersection = np.sum(manual_label_mask & auto_label_mask)
    vol_mask_manual = np.sum(manual_label_mask)
    vol_mask_auto = np.sum(auto_label_mask)

    denominator = vol_mask_manual + vol_mask_auto

    if denominator == 0:
        return np.nan

    return (2.0 * intersection) / denominator

def calculate_volume_ml(mask: np.ndarray, header, label: int) -> float:
    """
    Calculates the physical volume of a segmented region in milliliters (ml) 
    using voxel spatial dimensions extracted from the NIfTI header.
    """
    try:
        voxel_dims = header.get_zooms() 
        voxel_volume_mm3 = np.prod(voxel_dims)
        voxel_volume_ml = voxel_volume_mm3 / 1000.0 
    except Exception as e:
        print(f"[!] Warning: Cannot extract voxel dimensions from header: {e}. Defaulting to 1mm^3.")
        voxel_volume_ml = 1.0 / 1000.0 

    num_voxels = np.sum(mask == label)
    return float(num_voxels * voxel_volume_ml)

def compute_variation_percentage(pre_volume: float, mid_volume: float) -> float:
    """
    Calculates the percentage variation of the tumor/node volume between pre-RT and mid-RT scans.
    """
    if pre_volume == 0.0 and mid_volume == 0.0:
        return 0.0
    elif pre_volume == 0.0:
        return float('inf')
    else:
        return ((mid_volume - pre_volume) / pre_volume) * 100.0
