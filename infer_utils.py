# infer_utils.py — drop-in compatible with app.py
from __future__ import annotations
import io, os, json
from typing import List, Tuple, Optional, Dict, Iterable, Union

import numpy as np
import torch
import librosa
import scipy.signal as sig

# ====== Training-aligned constants ======
SR = 22050
N_FFT = 2048
HOP = 512
N_MFCC = 13
TARGET_FRAMES = 100

# Expose in the format app.py imports
AUDIO = {
    "SR": SR,
    "N_FFT": N_FFT,
    "HOP": HOP,
    "N_MFCC": N_MFCC,
    "TARGET_FRAMES": TARGET_FRAMES,
}

# ===== Audio I/O =====
def load_audio_bytes(data: bytes, sr: int = SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr

def load_wav(file_or_path: Union[str, bytes, io.BytesIO, "UploadedFile"], sr: int = SR) -> np.ndarray:
    """
    Accepts:
      - path string
      - raw bytes
      - BytesIO
      - Streamlit UploadedFile
    Returns mono float32 waveform at target sr.
    """
    if hasattr(file_or_path, "read"):  # Streamlit UploadedFile / file-like
        data = file_or_path.read()
        y, _ = load_audio_bytes(data, sr=sr)
        return y
    if isinstance(file_or_path, (bytes, bytearray)):
        y, _ = load_audio_bytes(file_or_path, sr=sr)
        return y
    if isinstance(file_or_path, io.BytesIO):
        y, _ = load_audio_bytes(file_or_path.getvalue(), sr=sr)
        return y
    # assume filesystem path
    y, _ = librosa.load(file_or_path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)

# ===== MFCC features (match preprocess) =====
def mfcc_13(y: np.ndarray, sr: int = SR) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    return m.astype(np.float32)  # (13, T)

def apply_mfcc_stats(mfcc: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is not None and std is not None and mean.shape == (N_MFCC,) and std.shape == (N_MFCC,):
        return (mfcc - mean[:, None]) / (std[:, None] + 1e-6)
    return (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)

def pad_or_crop_frames(arr: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] != N_MFCC:
        raise ValueError(f"Expected (13, T) MFCC, got {arr.shape}")
    T = arr.shape[1]
    if T == target_frames:
        return arr
    if T < target_frames:
        pad = np.zeros((arr.shape[0], target_frames - T), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)
    start = max(0, (T - target_frames)//2)
    return arr[:, start:start+target_frames]

def features_to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X[None, None, ...].astype(np.float32))  # (1,1,13,T)

# ===== Segmentation =====
def call_boundaries(y: np.ndarray, sr: int = SR,
                    fmin: int = 500, fmax: int = 3500,
                    n_fft: int = N_FFT, hop: int = HOP,
                    peak_distance: int = 6, k_mad: float = 3.0) -> List[Tuple[int,int]]:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= fmin) & (freqs <= fmax)
    env = S[band].mean(axis=0)
    env = sig.medfilt(env, kernel_size=9)
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-9
    thr = med + k_mad * mad
    peaks, _ = sig.find_peaks(env, height=thr, distance=peak_distance)
    spans: List[Tuple[int,int]] = []
    for p in peaks:
        left = max(0, p - 4)
        right = min(len(env) - 1, p + 4)
        spans.append((left, right))
    return spans

# app.py expects seconds
def segment_calls(y: np.ndarray, sr: int = SR) -> List[Tuple[float, float]]:
    spans = call_boundaries(y, sr)
    out: List[Tuple[float, float]] = []
    for l, r in spans:
        t0 = (l * HOP) / float(sr)
        t1 = (r * HOP) / float(sr)
        out.append((t0, t1))
    return out

def windows_fixed(y: np.ndarray, sr: int, window_s: float = 2.0, hop_s: float = 0.5) -> List[Tuple[float, float]]:
    n = len(y)
    win = max(1, int(window_s * sr))
    hop = max(1, int(hop_s * sr))
    spans: List[Tuple[float, float]] = []
    for s in range(0, max(1, n - win + 1), hop):
        e = min(n, s + win)
        spans.append((s / sr, e / sr))
    if not spans and n > 0:
        spans = [(0.0, n / sr)]
    return spans

# ===== Prob helpers & masking =====
def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()

def aggregate_predictions(probs_2d: np.ndarray, method: str = "logprob_max") -> Dict[int, float]:
    """
    probs_2d: [N_segments, C]
    returns {class_idx_in_kept: score}
    """
    if probs_2d.ndim != 2:
        raise ValueError("aggregate_predictions expects [N, C]")
    if method == "logprob_max":
        # robust to outliers: mean of log-probs, exponentiated (equiv to geometric mean)
        eps = 1e-9
        logps = np.log(np.clip(probs_2d, eps, 1.0))
        mean_logp = logps.mean(axis=0)                # [C]
        exps = np.exp(mean_logp - mean_logp.max())
        agg = exps / exps.sum()
    else:
        # simple max over segments
        agg = probs_2d.max(axis=0)
    return {i: float(v) for i, v in enumerate(agg)}

# ===== Species helpers =====
def pretty_species_name(scientific: str) -> str:
    s = scientific.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "classes" in data:
        return list(data["classes"])
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("class_names.json must be a list or a dict with key 'classes'")

def filter_species(class_names: List[str], include: Optional[Iterable[str]] = None,
                   exclude: Optional[Iterable[str]] = None) -> Tuple[List[int], List[str]]:
    include_set = set(x.strip() for x in include) if include else set(class_names)
    exclude_set = set(x.strip() for x in exclude) if exclude else set()
    kept = [i for i, n in enumerate(class_names) if (n in include_set) and (n not in exclude_set)]
    kept_names = [class_names[i] for i in kept]
    if not kept:
        # avoid empty selection; keep everything as fallback
        kept = list(range(len(class_names)))
        kept_names = list(class_names)
    return kept, kept_names

# ===== Model bundle & inference =====
class _Bundle:
    __slots__ = ("model", "device", "mfcc_mean", "mfcc_std")
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 mfcc_mean: Optional[np.ndarray], mfcc_std: Optional[np.ndarray]) -> None:
        self.model = model
        self.device = device
        self.mfcc_mean = mfcc_mean
        self.mfcc_std = mfcc_std

def _load_mfcc_stats(stats_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not stats_path or not os.path.exists(stats_path):
        return None, None
    with open(stats_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    mean = np.array(obj.get("mean"), dtype=np.float32) if "mean" in obj else None
    std  = np.array(obj.get("std"),  dtype=np.float32) if "std"  in obj else None
    if mean is not None and mean.shape != (N_MFCC,): mean = None
    if std  is not None and std.shape  != (N_MFCC,): std  = None
    return mean, std

def load_model(model_path: str, class_names_path: str, mfcc_stats_path: Optional[str] = None):
    """
    Returns (bundle, meta). Tries TorchScript first; falls back to torch.load().
    """
    device = torch.device("cpu")
    model = None
    # Try TorchScript
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    except Exception:
        # Try eager model or state-dict; if it's a whole nn.Module, this will work.
        obj = torch.load(model_path, map_location=device)
        if isinstance(obj, torch.nn.Module):
            model = obj.eval()
        else:
            raise RuntimeError(
                "Model is not a TorchScript module nor a full nn.Module. "
                "If you only saved state_dict, please load with your model definition."
            )
    mean, std = _load_mfcc_stats(mfcc_stats_path)
    bundle = _Bundle(model=model, device=device, mfcc_mean=mean, mfcc_std=std)
    meta = {"model_path": model_path, "device": str(device)}
    return bundle, meta

def wav_to_mfcc(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC, n_fft: int = N_FFT,
                hop_length: int = HOP, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    m = apply_mfcc_stats(m.astype(np.float32), None, None)  # per-clip if global not given here
    return pad_or_crop_frames(m, target_frames)

@torch.inference_mode()
def _predict_one(bundle: _Bundle, mfcc_13xT: np.ndarray) -> np.ndarray:
    X = features_to_tensor(mfcc_13xT).to(bundle.device)
    logits = bundle.model(X).squeeze(0)
    return logits_to_probs(logits)

def predict_mfcc_batch(bundle: _Bundle, mfcc_batch: np.ndarray) -> np.ndarray:
    """
    mfcc_batch: [N, 13, T]  -> probs [N, C]
    """
    out = []
    for i in range(mfcc_batch.shape[0]):
        m = mfcc_batch[i]
        # apply global stats if available
        m = apply_mfcc_stats(m, bundle.mfcc_mean, bundle.mfcc_std)
        m = pad_or_crop_frames(m, TARGET_FRAMES)
        out.append(_predict_one(bundle, m))
    return np.stack(out, axis=0)