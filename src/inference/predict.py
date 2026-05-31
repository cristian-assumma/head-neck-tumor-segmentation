import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import transform
from skimage.restoration import estimate_sigma
from tqdm import tqdm
import cv2

from mmengine.config import Config
from mmseg.apis import init_model, inference_model

# Import local modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.image_enhancement import apply_rician_denoise
from inference.postprocessing import apply_3d_morphological_closing

def preprocess_2d_slice(slice_2d: np.ndarray, final_size: tuple = (256, 256)) -> np.ndarray:
    """
    Applies the full preprocessing pipeline directly to a 2D array extracted from a NIfTI volume.
    """
    min_val, max_val = np.min(slice_2d), np.max(slice_2d)
    if min_val == max_val:
        norm_slice = np.zeros_like(slice_2d, dtype=np.uint8)
    else:
        norm_slice = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    img_norm = norm_slice.astype(np.float32) / 255.0
    sigma = estimate_sigma(img_norm, average_sigmas=True)
    
    # Rician Denoise (imported from preprocessing module)
    denoised = apply_rician_denoise(img_norm, sigma)
    denoised_uint8 = (denoised * 255).astype(np.uint8)

    # Median and Sobel
    median_filtered = cv2.medianBlur(denoised_uint8, 3)
    sobelx = cv2.Sobel(median_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(median_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    max_val_sobel = np.max(sobel_magnitude)

    if max_val_sobel == 0:
        edges = np.zeros_like(sobel_magnitude, dtype=np.uint8)
    else:
        sobel_magnitude = (sobel_magnitude / max_val_sobel) * 255
        sobel_magnitude = sobel_magnitude.astype(np.uint8)
        _, edges = cv2.threshold(sobel_magnitude, 50, 255, cv2.THRESH_BINARY)

    edges_mask = edges > 0
    enhanced = median_filtered.astype(np.float32)
    enhanced[edges_mask] *= 2.0
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Resize
    return cv2.resize(enhanced, final_size, interpolation=cv2.INTER_LINEAR)

def locate_nifti_file(patient_folder: str, suffix: str) -> str:
    for ext in ['.nii.gz', '.nii']:
        file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_{suffix}{ext}")
        if os.path.isfile(file_path):
            return file_path
    raise FileNotFoundError(f"NIfTI file with suffix '{suffix}' not found in {patient_folder}")

def infer_slice(slice_2d_array_256: np.ndarray, model_mmseg) -> np.ndarray:
    """Executes MMSegmentation inference on a preprocessed 2D slice."""
    img_rgb_like = np.stack([slice_2d_array_256] * 3, axis=-1)
    result = inference_model(model_mmseg, img_rgb_like)
    seg_logit = result.seg_logits.data.cpu()
    seg_pred_prob = F.softmax(seg_logit, dim=0)
    return torch.argmax(seg_pred_prob, dim=0).numpy().astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Run inference and 3D post-processing on MRI volumes.")
    parser.add_argument('--dataset-dir', type=str, default='../../data/Dataset_gz', help='Path to original NIfTI datasets')
    parser.add_argument('--excel-path', type=str, default='../../data/dataset_completo.xlsx', help='Path to dataset splits')
    parser.add_argument('--output-dir', type=str, default='../../data/Inference_Results', help='Output path')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained weights')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Initializing model on {device}...")
    
    cfg = Config.fromfile(args.config)
    cfg.device = device
    model = init_model(cfg, args.checkpoint, device=device)

    print("[*] Loading test patient cohort...")
    df_excel = pd.read_excel(args.excel_path, sheet_name='Test')
    test_patient_ids = df_excel['PatientID'].astype(str).tolist()

    # Morphological parameters
    min_sizes = {1: 50, 2: 1200}
    radii = {1: 1, 2: 3}

    for patient_id in tqdm(test_patient_ids, desc="Processing Patients"):
        patient_folder_path = os.path.join(args.dataset_dir, patient_id)
        output_patient_folder = os.path.join(args.output_dir, patient_id)
        os.makedirs(output_patient_folder, exist_ok=True)

        for timepoint in ["preRT", "midRT"]:
            try:
                nii_path = locate_nifti_file(patient_folder_path, f"{timepoint}_T2")
                vol_nifti = nib.load(nii_path)
                vol_data = vol_nifti.get_fdata()
                
                predicted_slices = []
                for i in range(vol_data.shape[2]):
                    slice_2d = vol_data[:, :, i]
                    original_shape = slice_2d.shape
                    
                    preprocessed_slice = preprocess_2d_slice(slice_2d)
                    pred_mask_256 = infer_slice(preprocessed_slice, model)
                    
                    pred_mask_orig = transform.resize(
                        pred_mask_256, original_shape, order=0, preserve_range=True, anti_aliasing=False
                    ).astype(np.uint8)
                    
                    predicted_slices.append(pred_mask_orig)

                raw_mask_3d = np.stack(predicted_slices, axis=2)
                post_mask_3d = apply_3d_morphological_closing(raw_mask_3d, min_sizes, radii)

                # Save finalized volume
                output_filename = f"SEG_{os.path.basename(nii_path).replace('.nii.gz', '_post.nii.gz').replace('.nii', '_post.nii')}"
                output_path = os.path.join(output_patient_folder, output_filename)
                
                nifti_img = nib.Nifti1Image(post_mask_3d, affine=vol_nifti.affine, header=vol_nifti.header)
                nib.save(nifti_img, output_path)

            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"\n[X] Error processing {patient_id} [{timepoint}]: {e}")

    print("\n[+] Inference and 3D Post-processing completed successfully.")

if __name__ == '__main__':
    main()
