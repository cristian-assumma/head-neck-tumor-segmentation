import os
import glob
import argparse
import warnings
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Local import
from metrics import compute_dice_coefficient, calculate_volume_ml, compute_variation_percentage

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation Performance (DSC, Volume Variation)')
    parser.add_argument('--dataset-dir', type=str, default='../../data/Dataset_gz', help='Path to Ground Truth NIfTI data')
    parser.add_argument('--inference-dir', type=str, default='../../data/Inference_Results', help='Path to automated predictions')
    parser.add_argument('--excel-path', type=str, default='../../data/dataset_completo.xlsx', help='Path to test patient list')
    parser.add_argument('--output-dir', type=str, default='../../results/Metrics', help='Directory to save metric reports')
    parser.add_argument('--use-postprocessed', action='store_true', default=True, help='Evaluate post-processed masks (_post)')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    labels_dict = {'GTVp': 1, 'GTVn': 2}
    auto_mask_suffix = "_post" if args.use_postprocessed else ""

    print(f"[*] Reading test set definitions from {args.excel_path}...")
    df_excel = pd.read_excel(args.excel_path, sheet_name='Test')
    test_patient_ids = df_excel['PatientID'].astype(str).tolist()

    results = []
    
    print("[*] Initiating Test Set Evaluation...")
    for patient_id in tqdm(test_patient_ids, desc="Evaluating Patients"):
        patient_gt_folder = os.path.join(args.dataset_dir, patient_id)
        patient_pred_folder = os.path.join(args.inference_dir, patient_id)
        patient_metrics = {'PatientID': patient_id}
        
        for timepoint in ['preRT', 'midRT']:
            try:
                # Ground Truth
                path_manual = glob.glob(os.path.join(patient_gt_folder, f"{patient_id}*{timepoint}_mask.nii*"))[0]
                manual_nib = nib.load(path_manual)
                manual_mask = manual_nib.get_fdata().astype(np.uint8)
                header = manual_nib.header
                
                # Prediction
                path_t2 = glob.glob(os.path.join(patient_gt_folder, f"{patient_id}*{timepoint}_T2.nii*"))[0]
                pred_filename = f"SEG_{os.path.basename(path_t2)}".replace(".nii.gz", f"{auto_mask_suffix}.nii.gz").replace(".nii", f"{auto_mask_suffix}.nii")
                path_auto = os.path.join(patient_pred_folder, pred_filename)
                auto_nib = nib.load(path_auto)
                auto_mask = auto_nib.get_fdata().astype(np.uint8)

                # Compute Metrics per class
                for label_name, label_id in labels_dict.items():
                    dsc = compute_dice_coefficient(manual_mask, auto_mask, label_id)
                    patient_metrics[f'DSC_{timepoint}_{label_name}'] = dsc
                    
                    vol_manual = calculate_volume_ml(manual_mask, header, label_id)
                    vol_auto = calculate_volume_ml(auto_mask, header, label_id)
                    
                    patient_metrics[f'Vol_{timepoint}_manual_{label_name}'] = vol_manual
                    patient_metrics[f'Vol_{timepoint}_auto_{label_name}'] = vol_auto

            except IndexError:
                # File not found
                for label_name in labels_dict.keys():
                    patient_metrics[f'DSC_{timepoint}_{label_name}'] = np.nan
                    patient_metrics[f'Vol_{timepoint}_manual_{label_name}'] = np.nan
                    patient_metrics[f'Vol_{timepoint}_auto_{label_name}'] = np.nan

        # Compute Volume Variations (Delta V)
        for label_name in labels_dict.keys():
            vol_pre_man = patient_metrics.get(f'Vol_preRT_manual_{label_name}', np.nan)
            vol_mid_man = patient_metrics.get(f'Vol_midRT_manual_{label_name}', np.nan)
            vol_pre_auto = patient_metrics.get(f'Vol_preRT_auto_{label_name}', np.nan)
            vol_mid_auto = patient_metrics.get(f'Vol_midRT_auto_{label_name}', np.nan)

            delta_v_man = compute_variation_percentage(vol_pre_man, vol_mid_man) if pd.notna(vol_pre_man) and pd.notna(vol_mid_man) else np.nan
            delta_v_auto = compute_variation_percentage(vol_pre_auto, vol_mid_auto) if pd.notna(vol_pre_auto) and pd.notna(vol_mid_auto) else np.nan
            
            patient_metrics[f'DeltaV_manual_{label_name}'] = delta_v_man
            patient_metrics[f'DeltaV_auto_{label_name}'] = delta_v_auto
            
            if pd.isna(delta_v_man) or pd.isna(delta_v_auto):
                patient_metrics[f'Error_DeltaV_{label_name}'] = np.nan
            else:
                patient_metrics[f'Error_DeltaV_{label_name}'] = abs(delta_v_auto - delta_v_man)

        results.append(patient_metrics)

    df_results = pd.DataFrame(results)
    df_results.to_excel(os.path.join(args.output_dir, 'detailed_metrics_per_patient.xlsx'), index=False)
    
    # Summary Statistics
    print("[*] Computing Summary Statistics...")
    summary = df_results.drop(columns=['PatientID']).mean().to_frame(name='Mean')
    summary['StdDev'] = df_results.drop(columns=['PatientID']).std()
    summary['Mean ± StdDev'] = summary.apply(lambda row: f"{row['Mean']:.3f} ± {row['StdDev']:.3f}" if pd.notna(row['Mean']) else "N/A", axis=1)
    
    summary.to_excel(os.path.join(args.output_dir, 'summary_metrics.xlsx'))
    print(summary[['Mean ± StdDev']])
    print(f"[+] Evaluation finished. Reports saved to {args.output_dir}")

if __name__ == '__main__':
    main()
