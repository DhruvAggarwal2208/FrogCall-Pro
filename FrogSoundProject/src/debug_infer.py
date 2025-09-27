# FrogSoundProject/src/debug_infer.py
import os, json, argparse, numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = os.path.join("FrogSoundProject","models","frog_cnn_best.pt")
DEFAULT_LABELS = os.path.join("FrogSoundProject","models","label_map.json")

# ---- Model that matches your good checkpoint ----
class FrogCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d((13, 1)),  # -> 64*13*1 (=832) features
        )
        # Indices mirror the checkpoint (linear layers at 2 and 5)
        self.classifier = nn.Sequential(
            nn.Flatten(),          # 0
            nn.Identity(),         # 1
            nn.Linear(64*13*1,256),# 2
            nn.ReLU(),             # 3
            nn.Identity(),         # 4
            nn.Linear(256,n_classes) # 5
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    # list
    if isinstance(lm, list) and all(isinstance(x,str) for x in lm):
        return lm
    # {"classes":[...]}
    if isinstance(lm, dict) and "classes" in lm:
        c = lm["classes"]
        if all(isinstance(x,str) for x in c): return c
        if all(isinstance(x,dict) and "name" in x for x in c): return [x["name"] for x in c]
    # name->idx
    if isinstance(lm, dict) and all(isinstance(k,str) for k in lm) and all(isinstance(v,int) for v in lm.values()):
        return [k for k,_ in sorted(lm.items(), key=lambda kv: kv[1])]
    # idx->name (keys str or int)
    if isinstance(lm, dict) and all(isinstance(v,str) for v in lm.values()):
        try:
            kv = [(int(k),v) for k,v in lm.items()]; kv.sort(key=lambda x:x[0]); return [v for _,v in kv]
        except: pass
    raise ValueError("Unrecognized label_map format.")

def wav_to_mfcc_13(wav_path, sr_target=16000):
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=160, center=True) # (13,T)
    return mfcc.astype(np.float32)

def pad_crop(a, T):
    c,t = a.shape
    if t < T:
        a = np.concatenate([a, np.zeros((c, T-t), dtype=a.dtype)], axis=1)
    elif t > T:
        a = a[:, :T]
    return a

def print_stats(tag, arr):
    arr = arr.astype(np.float64)
    print(f"{tag}: shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

def main():
    ap = argparse.ArgumentParser(description="Sanity-check single WAV against frog model.")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label-map", default=DEFAULT_LABELS)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--normalize", action="store_true", help="Per-coef z-norm (recommended).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== Debug Infer ===")
    print(f"WAV: {args.wav}")
    print(f"Model: {args.model}")
    print(f"Labels: {args.label_map}")
    print(f"Frames target: {args.frames} | Device: {args.device}")

    ckpt = torch.load(args.model, map_location="cpu")
    classes = None
    state_dict = None
    frames_ckpt = None
    if isinstance(ckpt, dict):
        if "classes" in ckpt and isinstance(ckpt["classes"], list):
            classes = ckpt["classes"]
            print(f"Loaded classes from checkpoint: {len(classes)}")
        if "frames" in ckpt and isinstance(ckpt["frames"], int):
            frames_ckpt = ckpt["frames"]
            print(f"Checkpoint frames: {frames_ckpt}")
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]

    if classes is None:
        classes = load_label_map(args.label_map)
        print(f"Loaded classes from label_map: {len(classes)}")

    T = frames_ckpt if frames_ckpt is not None else args.frames
    n_classes = len(classes)

    model = FrogCNN(n_classes=n_classes)
    if state_dict is None:
        # maybe raw state_dict
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict warnings -> missing: {list(missing)} | unexpected: {list(unexpected)}")
    device = torch.device(args.device)
    model.to(device).eval()

    # Audio -> MFCC -> (optional) z-norm -> pad/crop -> tensor
    mfcc = wav_to_mfcc_13(args.wav, sr_target=16000)
    print_stats("MFCC raw", mfcc)
    if args.normalize:
        mu = mfcc.mean(axis=1, keepdims=True)
        sd = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mu) / sd
        print_stats("MFCC z-norm", mfcc)
        print("Per-coef mean (first 5):", np.round(mu[:5,0],4))
        print("Per-coef std  (first 5):", np.round(sd[:5,0],4))
    mfcc = pad_crop(mfcc, T)
    print_stats(f"MFCC after pad/crop(T={T})", mfcc)

    x = torch.from_numpy(mfcc[None, None, ...]).to(device)  # (1,1,13,T)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    # Top-10
    idxs = np.argsort(-probs)[:10]
    print("\nTop-10 predictions:")
    for rank, i in enumerate(idxs, 1):
        print(f"{rank:2d}. {classes[i]:<28} {probs[i]*100:6.2f}%")

    # Quick sanity markers
    top_i = int(idxs[0])
    print("\nSUMMARY")
    print(f"Top class: {classes[top_i]} | confidence: {probs[top_i]*100:.2f}%")
    print(f"Sum(probs)={probs.sum():.6f} (should be 1.0)")
    print(f"argmax index: {top_i} / {n_classes}")

if __name__ == "__main__":
    main()
