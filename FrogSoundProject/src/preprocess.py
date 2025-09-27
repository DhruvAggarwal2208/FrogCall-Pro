# FrogSoundProject/src/preprocess_inat_mfcc.py
import os, sys, argparse, json, traceback
import numpy as np
import librosa

RAW_DIR = os.path.join("FrogSoundProject", "data", "raw", "inat_frogs")
OUT_DIR = os.path.join("FrogSoundProject", "data", "processed")

SR = 22050
N_MFCC = 13
MAX_TIME_STEPS = 100
MIN_SECONDS = 5.0

def mfcc_from_file(path: str):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if librosa.get_duration(y=y, sr=sr) < MIN_SECONDS:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # shape (n_mfcc, time)
    T = mfcc.shape[1]
    if T >= MAX_TIME_STEPS:
        mfcc = mfcc[:, :MAX_TIME_STEPS]
    else:
        pad = np.zeros((N_MFCC, MAX_TIME_STEPS - T), dtype=np.float32)
        mfcc = np.concatenate([mfcc.astype(np.float32), pad], axis=1)
    return mfcc.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Preprocess iNat frog audio to MFCC .npy files.")
    parser.add_argument("--raw_dir", default=RAW_DIR, help="Input root with species subfolders.")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output processed root.")
    parser.add_argument("--limit_per_species", type=int, default=0, help="Optional cap during preprocessing.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    errors = []
    total_ok = 0
    species_done = 0

    for species in sorted(os.listdir(args.raw_dir)):
        sp_in = os.path.join(args.raw_dir, species)
        if not os.path.isdir(sp_in):
            continue
        files = [f for f in os.listdir(sp_in) if f.lower().endswith((".wav",".mp3",".m4a",".ogg",".flac"))]
        if not files:
            continue

        sp_out = os.path.join(args.out_dir, species)
        os.makedirs(sp_out, exist_ok=True)

        kept = 0
        for fn in files:
            if args.limit_per_species and kept >= args.limit_per_species:
                break
            in_path = os.path.join(sp_in, fn)
            base, _ = os.path.splitext(fn)
            out_path = os.path.join(sp_out, base + ".npy")
            if os.path.exists(out_path):
                kept += 1
                continue
            try:
                feat = mfcc_from_file(in_path)
                if feat is None:
                    continue
                np.save(out_path, feat)
                kept += 1
                total_ok += 1
            except Exception as e:
                errors.append((in_path, str(e)))
        species_done += 1
        print(f"✅ {species}: {kept} files")

    print(f"\n🎯 Processed OK: {total_ok} files across {species_done} species")
    if errors:
        log = os.path.join(args.out_dir, "preprocess_errors.txt")
        with open(log, "w", encoding="utf-8") as f:
            for p, msg in errors:
                f.write(f"{p}\t{msg}\n")
        print(f"⚠️ Logged {len(errors)} failures to {log}")

if __name__ == "__main__":
    main()
