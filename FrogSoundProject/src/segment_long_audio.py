# -*- coding: utf-8 -*-
"""
Split raw audio into 5s chunks (non-overlapping by default), extract MFCC (13 x 100),
and save as .npy under data/processed/{species}/segment_XXXX.npy

Matches your train.py (sr=22050, n_mfcc=13, MAX_TIME_STEPS=100)
"""
import os
import re
import math
import librosa
import numpy as np
from tqdm import tqdm

RAW_ROOT = os.path.join("FrogSoundProject", "data", "raw")          # where your audio is
GUESS_SPECIES_FROM = "folder"                                       # "folder" or "filename"
PROCESSED_ROOT = os.path.join("FrogSoundProject", "data", "processed")

SR = 22050
N_MFCC = 13
WIN_SECS = 5.0
HOP_SECS = 5.0     # set <5.0 (e.g., 2.5) to create overlapping segments and get more data
MAX_TIME_STEPS = 100  # to match train.py (n_fft/hop below)

# MFCC params so that 5s -> ~100 frames
N_FFT = 1024
HOP_LENGTH = 1102   # ~ 22050 * 5 / 100
WIN_LENGTH = 1024

VALID_AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def species_from_path(filepath: str) -> str:
    if GUESS_SPECIES_FROM == "folder":
        # parent directory name as species (anuraset-like trees)
        return os.path.basename(os.path.dirname(filepath)).lower().replace(" ", "_")
    else:
        # try to parse a prefix_like_this__anything.ext
        base = os.path.splitext(os.path.basename(filepath))[0]
        m = re.match(r"([a-zA-Z]+_[a-zA-Z]+)", base.replace("-", "_"))
        return (m.group(1) if m else "unknown").lower()

def mfcc_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    # 13 x T -> pad/crop to 13 x 100
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        center=True
    ).astype(np.float32)

    # pad/crop time
    t = mfcc.shape[1]
    if t > MAX_TIME_STEPS:
        mfcc = mfcc[:, :MAX_TIME_STEPS]
    elif t < MAX_TIME_STEPS:
        pad = np.zeros((N_MFCC, MAX_TIME_STEPS - t), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=1)
    return mfcc

def main():
    safe_mkdir(PROCESSED_ROOT)

    # gather raw files
    raw_files = []
    for dp, _, files in os.walk(RAW_ROOT):
        for fn in files:
            if fn.lower().endswith(VALID_AUDIO_EXT):
                raw_files.append(os.path.join(dp, fn))

    if not raw_files:
        print(f"No audio found under {RAW_ROOT}")
        return

    created = 0
    skipped = 0
    for fp in tqdm(raw_files, desc="Segmenting"):
        try:
            sp = species_from_path(fp)
            if sp == "unknown":
                skipped += 1
                continue

            y, sr = librosa.load(fp, sr=SR, mono=True)
            if len(y) < int(WIN_SECS * sr):
                # too short for one full window
                skipped += 1
                continue

            step = int(HOP_SECS * sr)
            win = int(WIN_SECS * sr)
            out_dir = os.path.join(PROCESSED_ROOT, sp)
            safe_mkdir(out_dir)

            seg_idx = 0
            for start in range(0, len(y) - win + 1, step):
                seg = y[start:start+win]
                mfcc = mfcc_from_audio(seg, SR)
                out_path = os.path.join(out_dir, f"segment_{seg_idx:05d}.npy")
                np.save(out_path, mfcc)
                seg_idx += 1
                created += 1
        except Exception as e:
            # tolerate bad/odd files
            skipped += 1

    print(f"\n✅ Segmentation done. Created {created} MFCC segments. Skipped {skipped} items.")

if __name__ == "__main__":
    main()
