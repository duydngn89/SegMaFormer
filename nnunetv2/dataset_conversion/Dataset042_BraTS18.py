import os
import torch
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# -------------------- CHANGE THIS --------------------
pt_root = "/mnt/sata_disk/archive/BraTS2017_Training_Data"
# ------------------------------------------------------

task_id = 42
task_name = "BraTS2017_PT_Regions"
dataset_name = f"Dataset{task_id:03d}_{task_name}"

# nnUNet RAW PATH
from nnunetv2.paths import nnUNet_raw
out_base = join(nnUNet_raw, dataset_name)

imagesTr = join(out_base, "imagesTr")
labelsTr = join(out_base, "labelsTr")
maybe_mkdir_p(imagesTr)
maybe_mkdir_p(labelsTr)


def save_nifti(array, out_file):
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))  # fake spacing (required)
    sitk.WriteImage(img, out_file)


def onehot_to_multiclass(lbl):
    """
    lbl: shape (3, D, H, W)
    channels = [TC, WT, ET]
    """
    out = np.zeros(lbl.shape[1:], dtype=np.uint8)

    # WT must come first because WT contains TC+ET
    out[lbl[1] == 1] = 1   # WT
    out[lbl[0] == 1] = 2   # TC
    out[lbl[2] == 1] = 3   # ET

    return out



cases = sorted(os.listdir(pt_root))

for case in cases:
    case_dir = join(pt_root, case)
    if not os.path.isdir(case_dir):
        continue

    mod_file = join(case_dir, f"{case}_modalities.pt")
    lbl_file = join(case_dir, f"{case}_label.pt")

    

    if not os.path.exists(mod_file) or not os.path.exists(lbl_file):
        print(f"⚠ Skipping {case}, missing PT files")
        continue

    print(f"Converting {case} ...")

    # Load
    modalities = torch.load(mod_file, weights_only=False)
  # shape (4, D, H, W)
    labels = torch.load(lbl_file, weights_only=False)
     # shape (3, D, H, W)

    modalities = np.asarray(modalities)
    labels = np.asarray(labels)

    # Save each modality
    for idx in range(4):
        out_img = join(imagesTr, f"{case}_{idx:04d}.nii.gz")
        save_nifti(modalities[idx], out_img)

    # Convert one-hot → integer mask
    seg = onehot_to_multiclass(labels)

    seg_out = join(labelsTr, f"{case}.nii.gz")
    save_nifti(seg, seg_out)


# -------------------- CREATE DATASET.JSON --------------------
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

generate_dataset_json(
    out_base,
    channel_names={0: "Flair", 1: "T1", 2: "T1ce", 3: "T2"},
    labels={
        "background": 0,
        "whole tumor": (1, 2, 3),
        "tumor core": (2, 3),
        "enhancing tumor": (3,)
    },
    num_training_cases=len(cases),
    file_ending=".nii.gz",
    regions_class_order=(1, 2, 3),
    dataset_release="converted_from_pt"
)

print("\n====================================")
print("✅ Conversion Finished!")
print("📁 nnUNet dataset at:", out_base)
print("Run preprocessing:")
print(f"  nnUNetv2_plan_and_preprocess -d {task_id} -c 3d_fullres")
print("====================================")
