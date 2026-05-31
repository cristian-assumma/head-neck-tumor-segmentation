import os
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm

def create_patient_directories(base_path: str, split: str, patient_id: str):
    """
    Creates target subdirectories for storing extracted images and masks for a specific patient.
    """
    os.makedirs(os.path.join(base_path, split, "images", patient_id), exist_ok=True)
    os.makedirs(os.path.join(base_path, split, "masks", patient_id), exist_ok=True)

def normalize_intensity_to_uint8(image_volume: np.ndarray) -> np.ndarray:
    """
    Normalizes 3D MRI intensity values to the [0, 255] range and converts them to uint8.
    """
    min_val = np.min(image_volume)
    max_val = np.max(image_volume)
    if min_val == max_val:
        return np.zeros_like(image_volume, dtype=np.uint8)
    normalized = (image_volume - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)

def find_mask_slice_ranges(volume_mask: np.ndarray, axis: int = 2) -> list[tuple[int, int]]:
    """
    Identifies continuous ranges of axial slices containing labeled tumor structures.
    """
    num_slices = volume_mask.shape[axis]
    tumor_ranges = []
    in_tumor = False
    start_idx = None

    unique_values = np.unique(volume_mask)
    if len(unique_values) == 1 and unique_values[0] == 0:
        return []

    for i in range(num_slices):
        slice_mask = volume_mask.take(i, axis=axis)
        if np.any(slice_mask > 0):
            if not in_tumor:
                start_idx = i
                in_tumor = True
        else:
            if in_tumor:
                tumor_ranges.append((start_idx, i - 1))
                in_tumor = False

    if in_tumor:
        tumor_ranges.append((start_idx, num_slices - 1))

    return tumor_ranges

def save_slices_from_volume(volume_image: np.ndarray, volume_mask: np.ndarray, 
                            output_image_folder: str, output_mask_folder: str, 
                            axis: int = 2, step: int = 1, prefix: str = "volume", 
                            num_fixed_bg_slices: int = 5):
    """
    Extracts and saves targeted 2D axial slices containing tumors plus a contextual background buffer.
    """
    num_slices = volume_image.shape[axis]
    tumor_ranges = find_mask_slice_ranges(volume_mask, axis)

    if not tumor_ranges:
        print(f"[-] No labeled regions found for volume: {prefix}")
        return

    selected_slices = []
    for first_index, last_index in tumor_ranges:
        tumor_slices = list(range(first_index, last_index + 1, step))
        pre_tumor_slices = list(range(max(0, first_index - num_fixed_bg_slices), first_index))
        post_tumor_slices = list(range(last_index + 1, min(num_slices, last_index + num_fixed_bg_slices + 1)))
        selected_slices.extend(tumor_slices + pre_tumor_slices + post_tumor_slices)

    selected_slices = sorted(set(selected_slices))

    for i in selected_slices:
        slice_image = volume_image.take(i, axis=axis)
        slice_mask = volume_mask.take(i, axis=axis)

        normalized_image = normalize_intensity_to_uint8(slice_image)

        img_filename = f"{prefix}_slice{i:03d}.png"
        Image.fromarray(normalized_image).convert("L").save(os.path.join(output_image_folder, img_filename))
        Image.fromarray(slice_mask.astype(np.uint8)).convert("L").save(os.path.join(output_mask_folder, img_filename))

def locate_nifti_file(patient_folder: str, suffix: str) -> str:
    """
    Helper function to locate NIfTI files regardless of whether they are compressed (.nii.gz) or uncompressed (.nii).
    """
    for ext in ['.nii.gz', '.nii']:
        file_path = os.path.join(patient_folder, f"{os.path.basename(patient_folder)}_{suffix}{ext}")
        if os.path.isfile(file_path):
            return file_path
    raise FileNotFoundError(f"NIfTI file with suffix '{suffix}' not found in {patient_folder}")

def process_single_patient(patient_id: str, set_type: str, dataset_path: str, output_raw_folder: str):
    """
    Processes both preRT and midRT timepoints for a single patient, extracting axial slices.
    """
    patient_folder = os.path.join(dataset_path, patient_id)

    for timepoint in ["preRT", "midRT"]:
        patient_time_id = f"{patient_id}_{timepoint}"
        create_patient_directories(output_raw_folder, set_type, patient_time_id)
        try:
            file_image_path = locate_nifti_file(patient_folder, f"{timepoint}_T2")
            file_mask_path = locate_nifti_file(patient_folder, f"{timepoint}_mask")
            
            volume_image = nib.load(file_image_path).get_fdata()
            volume_mask = nib.load(file_mask_path).get_fdata()

            image_folder = os.path.join(output_raw_folder, set_type, "images", patient_time_id)
            mask_folder = os.path.join(output_raw_folder, set_type, "masks", patient_time_id)
            
            save_slices_from_volume(volume_image, volume_mask, image_folder, mask_folder, 
                                    axis=2, prefix=patient_time_id, num_fixed_bg_slices=5)
        except FileNotFoundError as e:
            print(f"[!] Missing file for patient {patient_id} [{timepoint}]: {e}. Skipping timepoint.")
        except Exception as e:
            print(f"[X] Error processing patient {patient_id} [{timepoint}]: {e}. Skipping timepoint.")

def extract_all_dataset_slices(base_path: str, dataset_excel_name: str, output_folder_name: str):
    """
    Orchestrates axial slice extraction across the stratified training and validation patient cohorts.
    """
    dataset_path = os.path.join(base_path, "Dataset_gz")
    output_raw_folder = os.path.join(base_path, output_folder_name)
    os.makedirs(output_raw_folder, exist_ok=True)

    dataset_file = os.path.join(base_path, dataset_excel_name)
    df = pd.read_excel(dataset_file, sheet_name=None)
    
    train_patients = df["Training"]["PatientID"].astype(str).tolist()
    val_patients = df["Validation"]["PatientID"].astype(str).tolist()

    print(f"Extracting training cohort slices ({len(train_patients)} patients)...")
    for patient_id in tqdm(train_patients, desc="Training Cohort Progress"):
        process_single_patient(patient_id, "train", dataset_path, output_raw_folder)

    print(f"Extracting validation cohort slices ({len(val_patients)} patients)...")
    for patient_id in tqdm(val_patients, desc="Validation Cohort Progress"):
        process_single_patient(patient_id, "val", dataset_path, output_raw_folder)

    print("[+] 2D axial slice extraction process finished successfully.")
