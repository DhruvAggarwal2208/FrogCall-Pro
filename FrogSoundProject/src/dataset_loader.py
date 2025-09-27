# FrogSoundProject/src/dataset_loader.py
import os
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Optional

PROCESSED_DIR = os.path.join("FrogSoundProject", "data", "processed")

def pad_or_crop(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    arr: (13, T)
    -> returns (13, target_len)
    """
    if arr.ndim != 2 or arr.shape[0] != 13:
        raise ValueError(f"Expected (13, T) array, got {arr.shape}")
    T = arr.shape[1]
    if T == target_len:
        return arr
    if T > target_len:
        return arr[:, :target_len]
    # pad with zeros at the end
    pad = np.zeros((arr.shape[0], target_len - T), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)

def scan_processed(root: str = PROCESSED_DIR) -> Dict[str, List[str]]:
    """
    Return {class_name: [file_paths...]} for .npy files.
    """
    by_class: Dict[str, List[str]] = {}
    for dirpath, _, files in os.walk(root):
        npys = [f for f in files if f.lower().endswith(".npy")]
        if not npys:
            continue
        label = os.path.basename(dirpath)
        by_class.setdefault(label, [])
        for f in npys:
            by_class[label].append(os.path.join(dirpath, f))
    return by_class

def load_dataset(
    processed_dir: str = PROCESSED_DIR,
    target_len: int = 100,
    min_frames: int = 20,            # skip ultra-short junk
    min_per_class: int = 1,          # include even 1-sample classes if you want
    max_per_class: Optional[int] = None,  # cap to balance; None = keep all
    shuffle_within_class: bool = True,
    seed: int = 42,
    add_channel_dim: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads MFCC .npy files from processed_dir/<class>/*.npy,
    normalizes lengths to (13, target_len), filters, and returns:
      X: (N, 13, target_len, 1) if add_channel_dim else (N, 13, target_len)
      y: (N,) integer labels
      classes: list of class names (index -> name)
    """
    rng = np.random.default_rng(seed)
    by_class = scan_processed(processed_dir)

    # Count raw before filtering
    raw_counts = {k: len(v) for k, v in by_class.items()}

    # Load, filter by min_frames, pad/crop, and optionally cap per class
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    classes = sorted([c for c in by_class.keys() if len(by_class[c]) >= min_per_class])
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    kept_counts = Counter()
    skipped_short = 0
    skipped_corrupt = 0
    skipped_cap = 0

    for cls in classes:
        files = by_class[cls]
        if shuffle_within_class:
            files = files.copy()
            rng.shuffle(files)
        take = max_per_class if max_per_class is not None else len(files)

        for fp in files:
            if max_per_class is not None and kept_counts[cls] >= take:
                skipped_cap += 1
                continue
            try:
                a = np.load(fp, mmap_mode="r")
                if a.ndim != 2 or a.shape[0] != 13:
                    skipped_corrupt += 1
                    continue
                if a.shape[1] < min_frames:
                    skipped_short += 1
                    continue
                a_std = pad_or_crop(np.asarray(a, dtype=np.float32), target_len)
                X_list.append(a_std)
                y_list.append(cls_to_idx[cls])
                kept_counts[cls] += 1
            except Exception:
                skipped_corrupt += 1
                continue

    if not X_list:
        raise RuntimeError("No usable samples found. Check processed directory and filters.")

    X = np.stack(X_list)  # (N, 13, target_len)
    if add_channel_dim:
        X = X[..., np.newaxis]  # (N, 13, target_len, 1)
    y = np.asarray(y_list, dtype=np.int64)

    # Summary
    print(f"📦 Classes discovered (raw)       : {len(raw_counts)}")
    print(f"   Non-empty after min_per_class : {len(classes)}")
    print(f"🧹 Skipped (too short)           : {skipped_short}")
    print(f"🧹 Skipped (corrupt/shape)       : {skipped_corrupt}")
    if max_per_class is not None:
        print(f"🧹 Skipped (over cap)            : {skipped_cap}")
    print(f"✅ Final dataset                  : X={X.shape}, y={y.shape}")
    if len(classes) > 20:
        print(f"   (showing first 20 classes) {classes[:20]}")
    else:
        print(f"   Classes: {classes}")

    return X, y, classes
