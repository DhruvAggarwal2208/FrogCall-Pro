import os, json, argparse, glob, numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Model (matches your trained checkpoint layout) ----------------
class FrogCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d((13, 1)),    # -> 64*13*1 = 832
        )
        # Indices chosen so the checkpoint lines up: linear at 2 and 5
        self.classifier = nn.Sequential(
            nn.Flatten(),            # 0
            nn.Identity(),           # 1
            nn.Linear(64*13*1, 256), # 2
            nn.ReLU(),               # 3
            nn.Identity(),           # 4
            nn.Linear(256, n_classes) # 5
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------- Utilities ----------------
def pad_or_crop(a, T):
    c, t = a.shape
    if t < T:
        pad = np.zeros((c, T - t), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=1)
    elif t > T:
        a = a[:, :T]
    return a

def try_find_processed_npy(wav_path, processed_root):
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    pattern = os.path.join(processed_root, "**", f"{stem}*.npy")
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None

def load_mfcc_stats(stats_path):
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        mean = np.asarray(js.get("mean"), dtype=np.float32)
        std  = np.asarray(js.get("std"),  dtype=np.float32)
        if mean.shape == (13,) and std.shape == (13,) and np.all(std > 0):
            return mean, std
    return None, None

def librosa_mfcc_13(wav_path, sr_target=16000, n_fft=1024, hop_length=160, win_length=None, center=True, htk=False, norm='ortho'):
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        center=center, htk=htk, norm=norm
    )  # (13, T)
    return mfcc.astype(np.float32)

def load_label_map_any(path, fallback=None):
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
        elif isinstance(lm, dict) and all(isinstance(k,str) for k in lm.keys()) and all(isinstance(v,int) for v in lm.values()):
            size = max(lm.values())+1
            cls = [None]*size
            for name, idx in lm.items():
                if 0 <= idx < size: cls[idx] = name
            if any(x is None for x in cls):
                cls = [k for k,_ in sorted(lm.items(), key=lambda kv: kv[1])]
        elif isinstance(lm, dict) and all(isinstance(v,str) for v in lm.values()):
            try:
                kv = [(int(k),v) for k,v in lm.items()]
                kv.sort(key=lambda x:x[0]); cls = [v for _,v in kv]
            except Exception:
                pass
    return cls if cls is not None else (fallback or [])

def window_rms(mfcc_win):
    return float(np.sqrt(np.mean(mfcc_win**2)))

# ---------------- Inference core ----------------
def infer_multi_windows(mfcc, model, frames, device, multi, strategy, focus, focus_k):
    T = frames
    total = mfcc.shape[1]
    if total <= T or multi <= 1:
        win = pad_or_crop(mfcc, T)[None, None, ...]
        with torch.no_grad():
            logits = model(torch.from_numpy(win).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    # evenly spaced starts
    starts = np.linspace(0, max(total - T, 0), num=multi, dtype=int)
    windows = []
    for s in starts:
        sub = mfcc[:, s:s+T]
        sub = pad_or_crop(sub, T)
        windows.append(sub)

    if focus == "loudest":
        scores = [window_rms(w) for w in windows]
        keep_idx = np.argsort(scores)[-min(focus_k, len(windows)):]
        windows = [windows[i] for i in keep_idx]

    agg_logits = None
    agg_probs  = None
    for sub in windows:
        x = sub[None, None, ...]
        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(device))
            probs  = F.softmax(logits, dim=1)

        if strategy in ("logit_mean", "logit-mean"):
            agg_logits = probs.new_zeros_like(logits) if agg_logits is None else agg_logits
            agg_logits += logits
        elif strategy == "max":
            agg_probs = probs if agg_probs is None else torch.maximum(agg_probs, probs)
        else:  # mean
            agg_probs = probs if agg_probs is None else agg_probs + probs

    if strategy in ("logit_mean", "logit-mean"):
        probs = F.softmax(agg_logits / len(windows), dim=1).cpu().numpy()[0]
    elif strategy == "max":
        probs = agg_probs.cpu().numpy()[0]
    else:
        probs = (agg_probs / len(windows)).cpu().numpy()[0]
    return probs

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Infer frog species from a WAV file.")
    ap.add_argument("--wav", required=True, help="Path to audio .wav")
    ap.add_argument("--npy", help="(Optional) Path to a processed MFCC .npy to force using training features")
    ap.add_argument("--processed-only", action="store_true", help="Require a processed .npy; error if none found")
    ap.add_argument("--model", default=os.path.join("FrogSoundProject","models","frog_cnn_best.pt"))
    ap.add_argument("--label-map", default=os.path.join("FrogSoundProject","models","label_map.json"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--uncertain-threshold", type=float, default=None, help="If top prob < thr, print 'UNCERTAIN'")
    ap.add_argument("--normalize", action="store_true", help="(fallback path only) per-coef z-norm per clip")
    ap.add_argument("--multi", type=int, default=12, help="# windows to evaluate")
    ap.add_argument("--strategy", choices=["mean","max","logit_mean","logit-mean"], default="mean", help="Aggregate across windows")
    ap.add_argument("--focus", choices=["none","loudest"], default="none", help="Optionally keep loudest windows only")
    ap.add_argument("--focus-k", type=int, default=6, help="How many windows to keep if --focus loudest")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--processed-root", default=os.path.join("FrogSoundProject","data","processed"))
    ap.add_argument("--mfcc-stats", default=os.path.join("FrogSoundProject","models","mfcc_stats.json"),
                    help="Optional global MFCC stats (mean/std per coef) for fallback librosa path")
    args = ap.parse_args()

    print("=== Frog Inference ===")
    print(f"WAV:      {args.wav}")
    print(f"Model:    {args.model}")
    print(f"Labels:   {args.label_map}")
    print(f"Frames:   {args.frames} | TopK: {args.topk}")

    # Load checkpoint
    ckpt = torch.load(args.model, map_location="cpu")
    classes = None
    target_frames = args.frames
    state_dict = None

    if isinstance(ckpt, dict):
        if "classes" in ckpt and isinstance(ckpt["classes"], list):
            classes = ckpt["classes"]; print(f"Loaded {len(classes)} classes from checkpoint.")
        if "frames" in ckpt and isinstance(ckpt["frames"], int):
            target_frames = ckpt["frames"]; print(f"Using target frames from checkpoint: {target_frames}")
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]

    if classes is None:
        classes = load_label_map_any(args.label_map)
        print(f"Loaded {len(classes)} classes from label_map.json")
    n_classes = len(classes)

    # Build model + load weights
    model = FrogCNN(n_classes=n_classes)
    if state_dict is None:
        if isinstance(ckpt, dict):
            tensor_only = {k:v for k,v in ckpt.items() if isinstance(v, torch.Tensor)}
            state_dict = tensor_only if tensor_only else ckpt
        else:
            state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ load_state_dict warnings — missing: {list(missing)}, unexpected: {list(unexpected)}")

    device = torch.device(args.device); model.to(device); model.eval()

    # ---------------- Feature source selection ----------------
    used_source = None
    used_path = None

    if args.npy:
        if not os.path.exists(args.npy):
            raise FileNotFoundError(f"--npy not found: {args.npy}")
        a = np.load(args.npy).astype(np.float32)
        if a.ndim == 3: a = a.squeeze(-1)
        if a.shape[0] != 13:
            raise ValueError(f"--npy has unexpected shape {a.shape}, expected (13,T) or (13,T,1)")
        used_source = "Processed (.npy, explicit)"; used_path = args.npy
        mfcc = a
    else:
        npy_match = try_find_processed_npy(args.wav, args.processed_root)
        if npy_match and os.path.exists(npy_match):
            a = np.load(npy_match).astype(np.float32)
            if a.ndim == 3: a = a.squeeze(-1)
            if a.shape[0] != 13:
                raise ValueError(f"Processed file has unexpected shape: {a.shape}")
            used_source = "Processed (.npy, auto)"; used_path = npy_match
            mfcc = a
        else:
            if args.processed_only:
                raise FileNotFoundError("No matching processed .npy found and --processed-only was set.")
            mfcc = librosa_mfcc_13(args.wav, sr_target=16000, n_fft=1024, hop_length=160, win_length=None, center=True, htk=False, norm='ortho')
            if args.normalize:
                mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
            mean, std = load_mfcc_stats(args.mfcc_stats)
            if mean is not None and std is not None:
                mfcc = (mfcc - mean[:,None]) / (std[:,None] + 1e-6)
            used_source = "Librosa MFCC (fallback)"; used_path = args.wav

    print(f"Feature source: {used_source}")
    print(f"Feature path:   {used_path}")

    # ---------------- Inference ----------------
    probs = infer_multi_windows(
        mfcc, model, target_frames, device,
        multi=max(1, args.multi), strategy=args.strategy,
        focus=args.focus, focus_k=args.focus_k
    )

    # Top-K
    topk = min(args.topk, n_classes)
    idxs = np.argsort(-probs)[:topk]
    print("\nTop predictions:")
    for i in idxs:
        print(f"  {classes[i]:<32} {probs[i]*100:6.2f}%")

    if args.uncertain_threshold is not None:
        top = probs[idxs[0]] * 100.0
        if top < args.uncertain_threshold:
            print(f"\n🤔 UNCERTAIN — top confidence {top:.2f}% (< {args.uncertain_threshold:.0f}%)")

if __name__ == "__main__":
    main()
