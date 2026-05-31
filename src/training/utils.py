import os
import numpy as np
from PIL import Image

def compute_class_weights(data_root: str, num_classes: int = 3, split: str = 'train', max_weight: float = 50.0) -> list[float]:
    """
    Calculates class weights to balance the composite loss function during training,
    handling extreme class imbalances (e.g., background > 98%).

    Parameters
    ----------
    data_root : str
        Base path of the preprocessed dataset.
    num_classes : int, optional
        Number of segmentation classes (default is 3).
    split : str, optional
        Dataset split to analyze, typically 'train' (default is 'train').
    max_weight : float, optional
        Maximum allowed weight to prevent gradient explosion for very rare classes (default is 50.0).

    Returns
    -------
    list[float]
        Normalized weights for each class.
    """
    print("[*] Computing class frequencies and weights...")
    class_counts = np.zeros(num_classes)
    mask_dir = os.path.join(data_root, split, 'masks')

    for patient_folder in sorted(os.listdir(mask_dir)):
        patient_path = os.path.join(mask_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for mask_file in sorted(os.listdir(patient_path)):
            mask_path = os.path.join(patient_path, mask_file)
            try:
                mask = np.array(Image.open(mask_path))
                for cls in range(num_classes):
                    class_counts[cls] += np.sum(mask == cls)
            except Exception as e:
                print(f"[!] Error loading {mask_path}: {e}")
                continue

    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    
    # Calculate inverse frequencies
    weights = 1.0 / (class_frequencies + 1e-6)
    
    # Clamp maximum weight to ensure stability
    weights = np.minimum(weights, max_weight)
    
    # Normalize so that the sum of weights equals num_classes
    weights = weights / weights.sum() * num_classes
    
    print(f"[+] Computed Class Weights: {weights}")
    return weights.tolist()
