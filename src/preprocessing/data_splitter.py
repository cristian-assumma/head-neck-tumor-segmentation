import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_clinical_data(gtvp_path: str, gtvn_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads clinical volume data from Excel files for both primary tumor (GTVp) and lymph nodes (GTVn).
    """
    gtvp_df = pd.read_excel(gtvp_path)
    gtvn_df = pd.read_excel(gtvn_path)
    return gtvp_df, gtvn_df

def merge_and_categorize_datasets(gtvp_df: pd.DataFrame, gtvn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges GTVp and GTVn datasets on PatientID and categorizes patients based on 
    tumor presence and volumetric size thresholds for stratified splitting.
    """
    merged_df = pd.merge(gtvp_df, gtvn_df, on="PatientID", suffixes=('_GTVp', '_GTVn'))

    # Tumor presence classification
    def classify_tumor_presence(row):
        has_gtvp = row["Pre-RT Volume (ml)_GTVp"] > 0
        has_gtvn = row["Pre-RT Volume (ml)_GTVn"] > 0

        if has_gtvp and has_gtvn:
            return "Both"
        elif has_gtvp:
            return "GTVp Only"
        elif has_gtvn:
            return "GTVn Only"
        else:
            return "None"

    merged_df["Tumor Type"] = merged_df.apply(classify_tumor_presence, axis=1)

    # Volumetric size classification based on calculated dataset medians
    def classify_size(volume, threshold):
        if volume == 0:
            return "None"
        elif volume <= threshold:
            return "Small"
        else:
            return "Large"

    # Dataset-specific median thresholds
    threshold_gtvp = 8.01
    threshold_gtvn = 8.89

    merged_df["Size_GTVp"] = merged_df["Pre-RT Volume (ml)_GTVp"].apply(lambda x: classify_size(x, threshold_gtvp))
    merged_df["Size_GTVn"] = merged_df["Pre-RT Volume (ml)_GTVn"].apply(lambda x: classify_size(x, threshold_gtvn))

    return merged_df

def split_dataset(merged_df: pd.DataFrame, test_size: float = 0.2, 
                  val_size: float = 0.25, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into balanced train, validation, and test sets using stratification 
    clinical categories (Tumor Type and Size classes).
    """
    # Stratified split to isolate the test set
    train_val, test = train_test_split(
        merged_df, 
        test_size=test_size, 
        stratify=merged_df[["Tumor Type", "Size_GTVp", "Size_GTVn"]], 
        random_state=random_seed
    )

    # Stratified split to separate train and validation sets from the remaining data
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, 
        test_size=adjusted_val_size, 
        stratify=train_val[["Tumor Type", "Size_GTVp", "Size_GTVn"]], 
        random_state=random_seed
    )

    return train, val, test

def save_stratified_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_path: str):
    """
    Saves the split datasets into a single Excel file with dedicated sheets.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        train.to_excel(writer, sheet_name="Training", index=False)
        val.to_excel(writer, sheet_name="Validation", index=False)
        test.to_excel(writer, sheet_name="Test", index=False)

def execute_data_split(base_path: str):
    """
    Orchestrator function to run the complete data splitting workflow.
    """
    gtvp_path = os.path.join(base_path, "GTVp_results.xlsx")
    gtvn_path = os.path.join(base_path, "GTVn_results.xlsx")
    output_path = os.path.join(base_path, "stratified_dataset.xlsx")

    print("Loading clinical data records...")
    gtvp_df, gtvn_df = load_clinical_data(gtvp_path, gtvn_path)
    
    print("Categorizing and merging datasets...")
    merged_df = merge_and_categorize_datasets(gtvp_df, gtvn_df)

    print("Performing stratified dataset split...")
    train, val, test = split_dataset(merged_df)

    print(f"Saving splits to: {output_path}")
    save_stratified_splits(train, val, test, output_path)
    print("Dataset stratification completed successfully.")
