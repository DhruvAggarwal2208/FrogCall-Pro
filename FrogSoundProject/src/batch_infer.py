import os, sys, csv, json, glob, argparse
from typing import List, Tuple, Optional
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Model (matches infer_wav.py) ----------------
class FrogCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d((13, 1)),  # -> 64 * 13 * 1
        )
        # keep linear at indices 2 and 5 to match your checkpoint
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 0
            nn.Identity(),                # 1
            nn.Linear(64*13*1, 256),      # 2
            nn.ReLU(),                    # 3
            nn.Identity(),                # 4
            nn.Linear(256, n_classes),    # 5
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------- Utils ----------------
def pad_or_crop(a: np.ndarray, T: int) -> np.ndarray:
    """a: (13, t) -> (13, T)"""
    c, t = a.shape
    if t < T:
        pad = np.zeros((c, T - t), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=1)
    elif t > T:
        a = a[:, :T]
    return a

def librosa_mfcc_13(wav_path: str, sr_target=16000, n_fft=1024, hop_length=160) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, center=True, norm="ortho")
    return mfcc.astype(np.float32)  # (13, T)

def try_find_processed_npy(wav_path: str, processed_root: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    hits = glob.glob(os.path.join(processed_root, "**", f"{stem}*.npy"), recursive=True)
    return hits[0] if hits else None

def load_mfcc_stats(stats_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        mean = np.asarray(js.get("mean"), dtype=np.float32)
        std  = np.asarray(js.get("std"),  dtype=np.float32)
        if mean.shape == (13,) and std.shape == (13,) and np.all(std > 0):
            return mean, std
    return None, None

def load_label_map_any(path: Optional[str], fallback=None) -> List[str]:
    cls = None
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        if isinstance(lm, list) and all(isinstance(x, str) for x in lm):
            cls = lm
        elif isinstance(lm, dict) and "classes" in lm and isinstance(lm["classes"], list):
            if all(isinstance(x, str) for x in lm["classes"]):
                cls = lm["classes"]
            elif all(isinstance(x, dict) and "name" in x for x in lm["classes"]):
                cls = [x["name"] for x in lm["classes"]]
        elif isinstance(lm, dict) and all(isinstance(k, str) for k in lm.keys()) and all(isinstance(v, int) for v in lm.values()):
            size = max(lm.values()) + 1
            cls = [None] * size
            for name, idx in lm.items():
                if 0 <= idx < size:
                    cls[idx] = name
            if any(x is None for x in cls):
                cls = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
        elif isinstance(lm, dict) and all(isinstance(v, str) for v in lm.values()):
            try:
                kv = [(int(k), v) for k, v in lm.items()]
                kv.sort(key=lambda x: x[0])
                cls = [v for _, v in kv]
            except Exception:
                pass
    return cls if cls is not None else (fallback or [])

def window_rms(mfcc_win: np.ndarray) -> float:
    return float(np.sqrt(np.mean(mfcc_win ** 2)))

def build_windows(mfcc: np.ndarray, T: int, multi: int) -> List[np.ndarray]:
    """Return list of (13, T) windows."""
    total = mfcc.shape[1]
    if total <= T or multi <= 1:
        return [pad_or_crop(mfcc, T)]
    starts = np.linspace(0, max(total - T, 0), num=multi, dtype=int)
    return [pad_or_crop(mfcc[:, s:s+T], T) for s in starts]

def model_probs_for_windows(model: nn.Module, device: torch.device, windows: List[np.ndarray]) -> np.ndarray:
    """
    windows: list of (13, T) np arrays
    returns: (N, C) softmax probabilities
    """
    if len(windows) == 0:
        return np.empty((0, 0), dtype=np.float32)
    x = torch.from_numpy(np.stack(windows).astype(np.float32))  # (N, 13, T)
    x = x.unsqueeze(1).to(device)                               # (N, 1, 13, T)
    with torch.no_grad():
        logits = model(x)                                       # (N, C)
        probs = F.softmax(logits, dim=1).cpu().numpy()          # (N, C)
    return probs

def aggregate_probs(perwin_probs: np.ndarray, strategy: str) -> np.ndarray:
    """
    perwin_probs: (N, C) softmax
    strategy: mean|max|logit_mean|logit-mean
    returns: (C,)
    """
    if perwin_probs.ndim != 2 or perwin_probs.size == 0:
        return perwin_probs
    if strategy == "max":
        return perwin_probs.max(axis=0)
    if strategy in ("logit_mean", "logit-mean"):
        eps = 1e-8
        logits = np.log(np.clip(perwin_probs, eps, 1.0))
        mean_logits = logits.mean(axis=0)
        exps = np.exp(mean_logits - mean_logits.max())
        return exps / exps.sum()
    # default mean
    return perwin_probs.mean(axis=0)

def apply_whitelist_mask(classes: List[str], whitelist_path: Optional[str]) -> Optional[np.ndarray]:
    """Return a boolean mask of size C with True for allowed classes, or None if no whitelist."""
    if not whitelist_path or not os.path.exists(whitelist_path):
        return None
    with open(whitelist_path, "r", encoding="utf-8") as f:
        allowed = {ln.strip() for ln in f if ln.strip()}
    mask = np.array([c in allowed for c in classes], dtype=bool)
    return mask

def apply_mask_and_renorm(probs: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return probs
    if mask.sum() == 0:
        return np.zeros_like(probs)
    p = probs.copy()
    p[~mask] = 0.0
    s = p.sum()
    return p if s <= 0 else p / s

def species_presence_from_windows(perwin_probs: np.ndarray, classes: List[str],
                                  threshold: float, min_windows: int,
                                  mask: Optional[np.ndarray]) -> List[str]:
    """
    perwin_probs: (N, C)
    Return list of species present in >= min_windows with per-window prob >= threshold,
    filtered by whitelist mask if provided.
    """
    if perwin_probs.size == 0:
        return []
    N, C = perwin_probs.shape
    if mask is not None:
        # zero out disallowed classes, then renormalize per window
        perwin_probs = perwin_probs * mask[np.newaxis, :]
        s = perwin_probs.sum(axis=1, keepdims=True)
        nz = (s > 0).astype(np.float32)
        perwin_probs = np.where(nz, perwin_probs / np.clip(s, 1e-8, None), perwin_probs)
    counts = (perwin_probs >= threshold).sum(axis=0)  # (C,)
    present_idx = np.where(counts >= min_windows)[0]
    return [classes[i] for i in present_idx]

# --------- Robust file discovery ----------
def list_audio_files(input_path: str, exts=(".wav",".mp3",".m4a",".flac")) -> List[str]:
    """Return audio files from a file or directory (recursive). Case-insensitive."""
    exts = tuple(e.lower() for e in exts)
    files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(exts):
            files.append(os.path.abspath(input_path))
    elif os.path.isdir(input_path):
        for root, _, fnames in os.walk(input_path):
            for fn in fnames:
                if fn.lower().endswith(exts):
                    files.append(os.path.abspath(os.path.join(root, fn)))
    return sorted(files)

# ---------------- Per-file inference ----------------
def infer_file(
    wav_path: str,
    model: nn.Module,
    device: torch.device,
    classes: List[str],
    frames: int,
    multi: int,
    strategy: str,
    focus: str,
    focus_k: int,
    processed_root: Optional[str],
    mfcc_stats_path: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Returns (agg_probs (C,), perwin_probs (N,C), feature_source)
    On failure: (None, None, reason)
    """
    feature_src = ""
    mfcc = None

    # Prefer processed MFCC if available
    if processed_root:
        npy = try_find_processed_npy(wav_path, processed_root)
        if npy and os.path.exists(npy):
            a = np.load(npy).astype(np.float32)
            if a.ndim == 3: a = a.squeeze(-1)
            if a.shape[0] == 13:
                mfcc = a
                feature_src = "Processed (.npy, auto)"

    if mfcc is None:
        # fallback librosa
        mfcc = librosa_mfcc_13(wav_path, sr_target=16000, n_fft=1024, hop_length=160)
        feature_src = "Librosa MFCC (fallback)"
        # optional global normalization
        mean, std = load_mfcc_stats(mfcc_stats_path)
        if mean is not None and std is not None:
            mfcc = (mfcc - mean[:, None]) / (std[:, None] + 1e-6)

    windows = build_windows(mfcc, frames, multi)

    # Optional focus on loudest windows
    if focus == "loudest" and len(windows) > 0:
        scores = [window_rms(w) for w in windows]
        keep_idx = np.argsort(scores)[-min(focus_k, len(windows)):]
        windows = [windows[i] for i in keep_idx]

    if len(windows) == 0:
        return None, None, "no_windows"

    # Per-window probs (N, C)
    perwin_probs = model_probs_for_windows(model, device, windows)

    # Aggregate across windows -> (C,)
    agg = aggregate_probs(perwin_probs, strategy)
    return agg, perwin_probs, feature_src

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch frog inference over a folder of audio files.")
    ap.add_argument("--input", required=True, help="Folder or a single audio file")
    ap.add_argument("--output", required=True, help="CSV path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--multi", type=int, default=60)
    ap.add_argument("--strategy", choices=["mean", "max", "logit_mean", "logit-mean"], default="logit_mean")
    ap.add_argument("--focus", choices=["none", "loudest"], default="loudest")
    ap.add_argument("--focus-k", type=int, default=15)
    ap.add_argument("--processed-root", default=os.path.join("FrogSoundProject", "data", "processed"))
    ap.add_argument("--mfcc-stats", default=os.path.join("FrogSoundProject", "models", "mfcc_stats.json"))
    ap.add_argument("--whitelist", default=None)
    ap.add_argument("--uncertain-threshold", type=float, default=70.0)
    ap.add_argument("--second-pass", action="store_true", help="Enable presence detection across windows")
    ap.add_argument("--presence-threshold", type=float, default=0.30, help="Per-window prob threshold for presence")
    ap.add_argument("--min-windows", type=int, default=3, help="#windows above threshold to mark presence")
    ap.add_argument("--alert-list", default=None)
    ap.add_argument("--include-exts", default="wav,WAV,mp3,m4a,flac", help="Comma-separated extensions to include")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load checkpoint + classes
    ckpt = torch.load(args.model, map_location="cpu")
    classes = None
    frames = args.frames
    state_dict = None
    if isinstance(ckpt, dict):
        if "classes" in ckpt and isinstance(ckpt["classes"], list):
            classes = ckpt["classes"]
        if "frames" in ckpt and isinstance(ckpt["frames"], int):
            frames = ckpt["frames"]
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
    if classes is None:
        classes = load_label_map_any(args.label_map)
    n_classes = len(classes)

    # Build & load model
    model = FrogCNN(n_classes=n_classes)
    if state_dict is None:
        if isinstance(ckpt, dict):
            tensor_only = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            state_dict = tensor_only if tensor_only else ckpt
        else:
            state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Whitelist / alert list
    wl_mask = apply_whitelist_mask(classes, args.whitelist)
    if args.whitelist and wl_mask is None:
        print(f"⚠️  Whitelist not found: {args.whitelist}")
    alert_set = set()
    if args.alert_list and os.path.exists(args.alert_list):
        with open(args.alert_list, "r", encoding="utf-8") as f:
            alert_set = {ln.strip() for ln in f if ln.strip()}
    elif args.alert_list:
        print(f"⚠️  Alert list not found: {args.alert_list}")

    # Collect inputs (multiple extensions, recursive)
    exts = tuple("." + e.lstrip(".").lower() for e in args.include_exts.split(",") if e.strip())
    inputs = list_audio_files(args.input, exts=exts)
    if not inputs:
        print("No audio files found.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["path", "error"])
        return

    print("=== Batch Frog Inference ===")
    print(f"Input:    {args.input}")
    print(f"Model:    {args.model}")
    print(f"Frames:   {frames}  | Device: {args.device}")
    print(f"Windows:  {args.multi}  | Strategy: {args.strategy}  | Focus: {args.focus} ({args.focus_k})")
    print(f"Whitelist: {args.whitelist}")
    print(f"Alert list: {args.alert_list}")
    print(f"Uncertain threshold: {args.uncertain_threshold:.1f}%  |  Presence: >= {args.presence_threshold} in >= {args.min_windows} windows\n")
    print(f"Found {len(inputs)} audio files.\n")

    # CSV header
    fields = [
        "path", "feature_src",
        "top1_name", "top1_prob",
        "top5", "uncertain",
        "present_species", "alert", "alert_hits",
        "error"
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()

        for idx, wav in enumerate(inputs, 1):
            print(f"[{idx}/{len(inputs)}] {wav}")
            row = {"path": wav, "error": ""}
            try:
                agg_probs, perwin_probs, feature_src = infer_file(
                    wav, model, device, classes, frames,
                    args.multi, args.strategy, args.focus, args.focus_k,
                    args.processed_root, args.mfcc_stats,
                )

                if agg_probs is None or agg_probs.size == 0:
                    row.update({"feature_src": feature_src, "error": "inference_failed"})
                    print("  ✖ Error: inference_failed")
                    writer.writerow(row)
                    continue

                # Apply whitelist mask
                agg_probs = apply_mask_and_renorm(agg_probs, wl_mask)

                # Top-k summary
                top_idx = np.argsort(-agg_probs)[:5]
                top_names = [classes[i] for i in top_idx]
                top_probs = [float(agg_probs[i]) for i in top_idx]
                row["feature_src"] = feature_src
                row["top1_name"] = top_names[0] if top_names else ""
                row["top1_prob"] = f"{top_probs[0]*100:.2f}%" if top_probs else ""
                row["top5"] = "; ".join(f"{n}:{p*100:.2f}%" for n, p in zip(top_names, top_probs))

                # Uncertain flag
                uncertain = (top_probs[0] * 100.0) < float(args.uncertain_threshold) if top_probs else True
                row["uncertain"] = "yes" if uncertain else "no"

                # Second-pass presence detection
                present_list = []
                if args.second_pass and perwin_probs is not None and perwin_probs.size > 0:
                    present_list = species_presence_from_windows(
                        perwin_probs, classes,
                        threshold=float(args.presence_threshold),
                        min_windows=int(args.min_windows),
                        mask=wl_mask
                    )
                row["present_species"] = "; ".join(present_list)

                # Alerts
                alert_hits = [sp for sp in present_list if sp in alert_set]
                row["alert"] = "yes" if alert_hits else "no"
                row["alert_hits"] = "; ".join(alert_hits)

                writer.writerow(row)

            except Exception as e:
                row["error"] = str(e)
                print(f"  ✖ Error: {e}")
                writer.writerow(row)

    print(f"\n✅ Saved: {args.output}\nDone.")

if __name__ == "__main__":
    main()
