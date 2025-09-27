# app.py — minimal Streamlit UI (only "Uncertain threshold")
# Works with: FrogSoundProject/models/frog_cnn_best.pt + label_map.json

import os, json, tempfile
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# -----------------------------
# Model (same head layout as training/infer_wav)
# -----------------------------
class FrogCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d((13, 1)),  # -> 64*13*1 = 832
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 0
            nn.Identity(),             # 1 (keeps ckpt indices aligned)
            nn.Linear(64*13*1, 256),   # 2
            nn.ReLU(),                 # 3
            nn.Identity(),             # 4 (placeholder)
            nn.Linear(256, n_classes)  # 5
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -----------------------------
# Utils
# -----------------------------
def load_label_map_flexible(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    if isinstance(lm, list) and all(isinstance(x, str) for x in lm):
        return lm

    if isinstance(lm, dict) and "classes" in lm:
        c = lm["classes"]
        if isinstance(c, list) and all(isinstance(x, str) for x in c):
            return c
        if isinstance(c, list) and all(isinstance(x, dict) and "name" in x for x in c):
            return [x["name"] for x in c]

    if isinstance(lm, dict) and all(isinstance(k, str) for k in lm.keys()) and all(isinstance(v, int) for v in lm.values()):
        size = max(lm.values()) + 1
        out = [None]*size
        for name, idx in lm.items():
            if 0 <= idx < size:
                out[idx] = name
        if any(x is None for x in out):
            out = [k for k,_ in sorted(lm.items(), key=lambda kv: kv[1])]
        return out

    if isinstance(lm, dict) and all(isinstance(v, str) for v in lm.values()):
        try:
            kv = [(int(k), v) for k, v in lm.items()]
            kv.sort(key=lambda x: x[0])
            return [v for _, v in kv]
        except Exception:
            pass

    for key in ("idx_to_class", "index_to_class", "id_to_label"):
        if isinstance(lm, dict) and key in lm and isinstance(lm[key], dict):
            kv = [(int(k), v) for k, v in lm[key].items()]
            kv.sort(key=lambda x: x[0])
            return [v for _, v in kv]

    for key in ("class_to_idx", "label_to_id"):
        if isinstance(lm, dict) and key in lm and isinstance(lm[key], dict):
            m = lm[key]
            size = max(m.values()) + 1
            out = [None]*size
            for name, idx in m.items():
                if 0 <= idx < size:
                    out[idx] = str(name)
            return [c if c is not None else f"class_{i}" for i, c in enumerate(out)]

    # last resort: newline list
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) >= 2 and all("," not in ln for ln in lines):
            return lines
    except Exception:
        pass

    raise ValueError("label_map format not recognized")

def wav_to_mfcc_13(wav_path: str, sr_target: int = 16000, normalize: bool = True) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=160, center=True)
    if normalize:
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
    return mfcc.astype(np.float32)

def pad_or_crop(a: np.ndarray, T: int) -> np.ndarray:
    c, t = a.shape
    if t < T:
        pad = np.zeros((c, T-t), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=1)
    elif t > T:
        a = a[:, :T]
    return a

def evenly_spaced_windows(total_frames: int, window_frames: int, n: int):
    if total_frames <= window_frames or n <= 1:
        return [(0, min(window_frames, total_frames))]
    span = total_frames - window_frames
    starts = [int(round(i * span / (n - 1))) for i in range(n)]
    return [(s, s + window_frames) for s in starts]

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_meta(model_path: str, label_map_path: str, device: str = "cpu"):
    ckpt = torch.load(model_path, map_location="cpu")

    classes = None
    frames = None
    state_dict = None
    warns = []

    if isinstance(ckpt, dict):
        if "classes" in ckpt and isinstance(ckpt["classes"], list):
            classes = ckpt["classes"]
        if "frames" in ckpt and isinstance(ckpt["frames"], int):
            frames = ckpt["frames"]

        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            # raw state_dict fallback
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt

    if classes is None:
        classes = load_label_map_flexible(label_map_path)

    model = FrogCNN(n_classes=len(classes))
    if state_dict is None:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warns.append(f"Missing keys: {sorted(missing)}")
    if unexpected:
        warns.append(f"Unexpected keys: {sorted(unexpected)}")

    model.to(torch.device(device))
    model.eval()
    return model, classes, frames, ("\n".join(warns) if warns else None)

def predict_aggregate(model, classes, wav_path, frames, windows=12, strategy="mean", device="cpu"):
    mfcc = wav_to_mfcc_13(wav_path, 16000, True)
    T = mfcc.shape[1]
    idxs = evenly_spaced_windows(T, frames, windows)
    per_win = []
    with torch.no_grad():
        for (s, e) in idxs:
            chunk = pad_or_crop(mfcc[:, s:e], frames)
            x = torch.from_numpy(chunk[None, None, ...]).to(device)
            p = F.softmax(model(x), dim=1).cpu().numpy()[0]
            per_win.append(p)
    per_win = np.stack(per_win, axis=0)

    if strategy == "max":
        probs = per_win.max(axis=0)
    elif strategy in ("logit_mean", "logit-mean"):
        logp = np.log(per_win + 1e-12)
        logits = logp - logp.mean(axis=1, keepdims=True)
        logits = logits.mean(axis=0)
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
    else:
        probs = per_win.mean(axis=0)

    return probs, per_win

# -----------------------------
# UI (minimal)
# -----------------------------
st.set_page_config(page_title="Frog Classifier", page_icon="🐸", layout="centered")
st.title("🐸 Frog Classifier")
st.caption("Upload a WAV to classify frog species. Only one setting: uncertainty threshold.")

# Resolve default paths relative to this file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(APP_DIR, "..", "models", "frog_cnn_best.pt"))
LABEL_MAP_PATH = os.path.normpath(os.path.join(APP_DIR, "..", "models", "label_map.json"))

# Sidebar: only uncertainty threshold
with st.sidebar:
    st.header("Settings")
    uncertain_th = st.slider("Uncertain threshold (top prob)", 0.0, 1.0, 0.0, 0.01)

# Load model once
device_choice = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found:\n{MODEL_PATH}")
        st.stop()
    if not os.path.exists(LABEL_MAP_PATH):
        st.error(f"Label map not found:\n{LABEL_MAP_PATH}")
        st.stop()

    model, classes, ckpt_frames, warn = load_model_and_meta(MODEL_PATH, LABEL_MAP_PATH, device=device_choice)
    frames = ckpt_frames if isinstance(ckpt_frames, int) else 100
    if warn:
        st.info(warn)
    st.caption(f"Using frames: **{frames}** | Device: **{device_choice}**")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File input
uploaded = st.file_uploader("Upload WAV", type=["wav"])
wav_path: Optional[str] = None
if uploaded is not None:
    tmp_path = os.path.join(tempfile.gettempdir(), "frog_tmp_upload.wav")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    wav_path = tmp_path

run = st.button("🔍 Run Inference", type="primary", disabled=wav_path is None)

if run and wav_path:
    try:
        # Fixed inference settings (no extra UI):
        # - multi windows = 12
        # - aggregation = mean
        probs, per_window = predict_aggregate(
            model, classes, wav_path, frames, windows=12, strategy="mean", device=device_choice
        )
        topk = 5
        idxs = np.argsort(-probs)[:topk]

        st.subheader("Top Predictions")
        if uncertain_th > 0.0 and probs[idxs[0]] < uncertain_th:
            st.warning(f"🤔 UNCERTAIN — top confidence {probs[idxs[0]]*100:.2f}% is below {uncertain_th*100:.0f}%")

        for i in idxs:
            p = float(probs[i])
            st.write(f"**{classes[i]}** — {p*100:.2f}%")
            st.progress(min(max(p, 0.0), 1.0))

        # Tiny diagnostic chart (best class per window)
        best_per_window = per_window.max(axis=1)
        st.line_chart(best_per_window)

    except Exception as e:
        st.error(f"Inference failed: {e}")