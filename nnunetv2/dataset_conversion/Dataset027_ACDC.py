import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    nifti_files, join, maybe_mkdir_p, save_json
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


# -------------------------------------------------------------
# Make directories
# -------------------------------------------------------------
def make_out_dirs(dataset_id: int, task_name="ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    imagesTr = out_dir / "imagesTr"
    labelsTr = out_dir / "labelsTr"
    imagesTs = out_dir / "imagesTs"
    labelsTs = out_dir / "labelsTs"

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    return out_dir, imagesTr, labelsTr, imagesTs, labelsTs


# -------------------------------------------------------------
# Create 5-fold split (standard for ACDC)
# -------------------------------------------------------------
def create_ACDC_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
    nii_files = nifti_files(labelsTr_folder, join=False)
    patients = np.unique([i[:len('patient000')] for i in nii_files])
    rs = np.random.RandomState(seed)
    rs.shuffle(patients)

    splits = []
    for fold in range(5):
        val_patients = patients[fold::5]
        train_patients = [i for i in patients if i not in val_patients]

        val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
        train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]

        splits.append({'train': train_cases, 'val': val_cases})

    return splits


# -------------------------------------------------------------
# Copy ACDC images + labels
# -------------------------------------------------------------
def copy_files(src_data_folder: Path, imagesTr: Path, labelsTr: Path, imagesTs: Path, labelsTs: Path):
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0

    # -----------------------------
    # copy TRAINING files
    # -----------------------------
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():

            # image
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, imagesTr / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1

            # label
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labelsTr / file.name.replace("_gt", ""))

    # -----------------------------
    # copy TEST files (images + GT)
    # -----------------------------
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():

            # image
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, imagesTs / f"{file.stem.split('.')[0]}_0000.nii.gz")

            # label (if exists in your dataset)
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labelsTs / file.name.replace("_gt", ""))

    return num_training_cases


# -------------------------------------------------------------
# MAIN conversion logic
# -------------------------------------------------------------
def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir, imagesTr, labelsTr, imagesTs, labelsTs = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), imagesTr, labelsTr, imagesTs, labelsTs)

    generate_dataset_json(
        dataset_path=str(out_dir),
        channel_names={0: "cineMRI"},
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str,
        help="ACDC dataset folder with 'training' and 'testing'"
    )
    parser.add_argument(
        "-d", "--dataset_id", type=int, default=27,
        help="nnU-Net Dataset ID, default=27"
    )
    args = parser.parse_args()

    print("Converting...")
    convert_acdc(args.input_folder, args.dataset_id)

    # Generate splits
    dataset_name = f"Dataset{args.dataset_id:03d}_ACDC"
    labelsTr = join(nnUNet_raw, dataset_name, 'labelsTr')
    preprocessed_dir = join(nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(preprocessed_dir)

    split = create_ACDC_split(labelsTr)
    save_json(split, join(preprocessed_dir, 'splits_final.json'), sort_keys=False)

    print("Done!")
