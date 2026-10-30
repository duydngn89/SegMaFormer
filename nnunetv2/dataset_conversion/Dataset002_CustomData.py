#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

REAL_MASK_ROOT = Path("/media/duydnguyen/DATA3/AbdomenAtlas1.0/Masks")

# ============================================================
# BUILD DATASET JSON
# ============================================================
def build_dataset_json(dataset_dir: Path, dataset_name: str):
    labels = sorted((dataset_dir / "labelsTr").glob("*.nii.gz"))
    num_training = len(labels)

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names={0: "CT"},
        labels={
            "background": 0,
            "bones": 1,
            "kidney_left": 2,
            "kidney_right": 3,
            "urinary_bladder": 4,
            "pancreas": 5,
            "gallbladder": 6,
            "aorta": 7,
            "inferior_vena_cava": 8,
        },
        num_training_cases=num_training,
        file_ending=".nii.gz",
        dataset_name=dataset_name,
        overwrite_image_reader_writer="NibabelIOWithReorient"
    ) 

# ============================================================
# BONE LIST (Full coverage)
# ============================================================
BONE_FILES = {
    # Vertebrae
    "vertebrae_S1.nii.gz",
    *[f"vertebrae_L{i}.nii.gz" for i in range(1, 6)],
    *[f"vertebrae_T{i}.nii.gz" for i in range(1, 13)],
    *[f"vertebrae_C{i}.nii.gz" for i in range(1, 8)],

    # Sacrum (Important!)
    "sacrum.nii.gz",

    # Ribs
    *[f"rib_left_{i}.nii.gz" for i in range(1, 13)],
    *[f"rib_right_{i}.nii.gz" for i in range(1, 13)],

    # Skull
    "skull.nii.gz",
    "mandible.nii.gz",

    # Shoulder Girdle
    "clavicula_left.nii.gz",
    "clavicula_right.nii.gz",
    "scapula_left.nii.gz",
    "scapula_right.nii.gz",

    # Arms
    "humerus_left.nii.gz",
    "humerus_right.nii.gz",
    "radius_left.nii.gz",
    "radius_right.nii.gz",
    "ulna_left.nii.gz",
    "ulna_right.nii.gz",

    # Pelvis
    "hip_left.nii.gz",
    "hip_right.nii.gz",

    # Legs
    "femur_left.nii.gz",
    "femur_right.nii.gz",
    "tibia_left.nii.gz",
    "tibia_right.nii.gz",
    "fibula_left.nii.gz",
    "fibula_right.nii.gz",
    "patella_left.nii.gz",
    "patella_right.nii.gz",

    # Sternum
    "sternum.nii.gz",
    "costal_cartilages.nii.gz",
}


# Organs
KIDNEY_LEFT = "kidney_left.nii.gz"
KIDNEY_RIGHT = "kidney_right.nii.gz"
BLADDER_FILE = "urinary_bladder.nii.gz"

PANCREAS_FILE = "pancreas.nii.gz"

GALLBLADDER_FILE = "gallbladder.nii.gz"
AORTA_FILE = "aorta.nii.gz"
POSTCANVA_FILE = "inferior_vena_cava.nii.gz"

LABEL_MAP = {
    0: "background",
    1: "bones",
    2: "kidney_left",
    3: "kidney_right",
    4: "urinary_bladder",
    5: "pancreas",
    6: "gallbladder",
    7: "aorta",
    8: "inferior_vena_cava",
}

#============================================================
# UNION MERGE: PSEUDO ∪ REAL
# ============================================================
def union_merge_real_and_pseudo(real_mask_path: Path, combined: np.ndarray):
    """
    UNION merge:
        final = pseudo ∪ real

    - If pseudo has organ → keep
    - If real has organ → add
    - Never remove pseudo organs
    - Bones remain highest-priority
    """

    if not real_mask_path.exists():
        print(f"Real mask not found: {real_mask_path}. Skipping real organ merge.")
        return combined

    print(f" Merging real organs from: {real_mask_path.name}")
    real = nib.load(str(real_mask_path)).get_fdata().astype(np.int16)

    # real_class → final_label
    mapping = {
        2: 6,    # gallbladder
        6: 5,    # pancreas
        3: 2,    # kidney_left
        4: 3,    # kidney_right
        12: 4,
        1: 7,
        7: 8  # postcava
    }

    for real_cls, out_lbl in mapping.items():
        mask = (real == real_cls)
        combined[mask] = out_lbl

    return combined


# ============================================================
# COMBINE PSEUDO MASKS + UNION REAL ORGANS
# ============================================================
def combine_and_remap(mask_dir: Path,
                      out_label_path: Path,
                      success_log: Path,
                      ref_img: nib.Nifti1Image,
                      real_mask_path: Path = None):

    shape = ref_img.shape
    combined = np.zeros(shape, dtype=np.uint8)

    def write_label(mask_path: Path, label: int):
        nonlocal combined
        if not mask_path.exists():
            return
        arr = nib.load(str(mask_path)).get_fdata() > 0
        combined[arr] = label

    # -------------------------
    # 1) PSEUDO organs
    # -------------------------
    
    write_label(mask_dir / GALLBLADDER_FILE, 6)
    write_label(mask_dir / PANCREAS_FILE, 5)
    write_label(mask_dir / BLADDER_FILE, 4)
    write_label(mask_dir / KIDNEY_LEFT, 2)
    write_label(mask_dir / KIDNEY_RIGHT, 3)

    write_label(mask_dir / AORTA_FILE, 7)
    write_label(mask_dir / POSTCANVA_FILE, 8)
    combined = union_merge_real_and_pseudo(real_mask_path, combined)
    # -------------------------
    # 2) Bones (highest priority)
    # -------------------------
    for bf in BONE_FILES:
        write_label(mask_dir / bf, 1)

    # Save final mask
    out_nii = nib.Nifti1Image(combined, ref_img.affine, ref_img.header)
    nib.save(out_nii, str(out_label_path))

    # Stats logging
    unique, counts = np.unique(combined, return_counts=True)
    stats = {LABEL_MAP.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}

    with open(success_log, "a") as f:
        f.write(f"{mask_dir.name}: {stats}\n")


# ============================================================
# PROCESS ONE CASE
# ============================================================
def process_case(case_file: Path, ct_root: Path, mask_root: Path,
                 imagesTr: Path, labelsTr: Path,
                 log_path: Path, success_log: Path):

    try:
        case_id = case_file.name.replace(".nii.gz", "")
        ct_path = ct_root / f"{case_id}.nii.gz"
        mask_dir = mask_root / case_id
        real_mask_path = REAL_MASK_ROOT / f"{case_id}.nii.gz"

        if not ct_path.exists():
            raise RuntimeError(f"Missing CT file: {ct_path}")
        if not mask_dir.exists():
            raise RuntimeError(f"Missing pseudo mask directory: {mask_dir}")

        out_mask = labelsTr / f"{case_id}.nii.gz"

        if out_mask.exists():
            msg = f"[SKIP EXISTS] {case_id}"
            print(msg)
            with open(success_log, "a") as f:
                f.write(msg + "\n")
            return

        print(f"[INFO] Processing {case_id}")

        ct_img = nib.load(str(ct_path))

        # Generate combined mask
        combine_and_remap(mask_dir, out_mask, success_log, ct_img, real_mask_path)

        # Save CT to imagesTr
        img_out_path = imagesTr / f"{case_id}_0000.nii.gz"
        img_out_path.write_bytes(ct_path.read_bytes())

    except Exception as e:
        err_msg = f"[ERROR] {case_file.name}: {e}"
        print(err_msg)
        with open(log_path, "a") as f:
            f.write(err_msg + "\n")
            f.write(traceback.format_exc() + "\n")
        with open(success_log, "a") as f:
            f.write(f"[SKIP ERROR] {case_file.name}\n")


# ============================================================
# MAIN SCRIPT
# ============================================================
# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--dataset_id", type=int, required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--test_fraction", type=float, default=0.10)  # 10% test
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    input_root = Path(args.input_root)
    ct_root = input_root / "Images"
    pseudo_root = input_root / "result"

    dataset_dir = Path(args.output_root) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTs = dataset_dir / "labelsTs"   # optional but useful if you want to evaluate

    for d in (imagesTr, labelsTr, imagesTs, labelsTs):
        d.mkdir(parents=True, exist_ok=True)

    log_path = dataset_dir / "error_log.txt"
    success_log = dataset_dir / "success_log.txt"

    # reset logs
    for f in (log_path, success_log):
        if f.exists():
            f.unlink()

    case_files = sorted(ct_root.glob("BDMAP_*.nii.gz"))
    if len(case_files) == 0:
        raise RuntimeError(f"No cases found in {ct_root} matching BDMAP_*.nii.gz")

    # --- split 90/10 ---
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(case_files))
    rng.shuffle(idx)

    n_test = max(1, int(round(args.test_fraction * len(case_files))))
    test_set = set(idx[:n_test])

    train_cases = [case_files[i] for i in range(len(case_files)) if i not in test_set]
    test_cases  = [case_files[i] for i in range(len(case_files)) if i in test_set]

    print(f"[SPLIT] total={len(case_files)} train={len(train_cases)} test={len(test_cases)} "
          f"(test_fraction={args.test_fraction}, seed={args.seed})")

    # process train + test
    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        futures = {}

        for ct_file in train_cases:
            futures[ex.submit(
                process_case, ct_file, ct_root, pseudo_root,
                imagesTr, labelsTr, log_path, success_log
            )] = ct_file

        for ct_file in test_cases:
            futures[ex.submit(
                process_case, ct_file, ct_root, pseudo_root,
                imagesTs, labelsTs, log_path, success_log
            )] = ct_file

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] Crash: {e}")

    # dataset.json uses training count (labelsTr)
    build_dataset_json(dataset_dir, args.dataset_name)


if __name__ == "__main__":
    main()