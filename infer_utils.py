from __future__ import annotations
import io
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import librosa
import scipy.signal as sig

# ====== Training-aligned constants ======
SR = 22050            # <- matches preprocess.py
N_FFT = 2048          # librosa default (explicit here for clarity)
HOP = 512             # librosa default hop
N_MFCC = 13
TARGET_FRAMES = 100   # fixed length used in training

# ===== Audio I/O =====
def load_audio_bytes(data: bytes, sr: int = SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr

# ===== MFCC features (match preprocess) =====
def mfcc_13(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Use librosa defaults (n_fft=2048, hop_length=512) and 13 coefficients.
    """
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return m.astype(np.float32)  # (13, T)

def apply_mfcc_stats(mfcc: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    """
    Global mean/std if provided (mfcc_stats.json); else per-clip z-norm.
    """
    if mean is not None and std is not None and mean.shape == (N_MFCC,) and std.shape == (N_MFCC,):
        return (mfcc - mean[:, None]) / (std[:, None] + 1e-6)
    return (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)

def pad_or_crop_frames(arr: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    arr: (13, T) -> (13, target_frames) by center-crop or right-pad.
    """
    if arr.ndim != 2 or arr.shape[0] != N_MFCC:
        raise ValueError(f"Expected (13, T) MFCC, got {arr.shape}")
    T = arr.shape[1]
    if T == target_frames:
        return arr
    if T < target_frames:
        pad = np.zeros((arr.shape[0], target_frames - T), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)
    # crop (center)
    start = max(0, (T - target_frames)//2)
    return arr[:, start:start+target_frames]

def features_to_tensor(X: np.ndarray) -> torch.Tensor:
    """(13, T) -> (1,1,13,T)"""
    return torch.from_numpy(X[None, None, ...].astype(np.float32))

# ===== Segmentation for 'calls' path (same STFT grid as features) =====
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

# ===== Prob helpers & masking =====
def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()

def mask_probs_with_whitelist(probs: np.ndarray, class_names: List[str], whitelist: set[str]) -> np.ndarray:
    p = probs.copy().astype(float)
    mask = np.array([name in whitelist for name in class_names], dtype=bool)
    if p.ndim == 1:
        if not mask.any(): return np.zeros_like(p)
        p[~mask] = 0.0
        s = p.sum()
        return p / s if s > 0 else p
    if not mask.any(): return np.zeros_like(p)
    p[:, ~mask] = 0.0
    sums = p.sum(axis=1, keepdims=True)
    nz = sums > 0
    p[nz] /= sums[nz]
    return p

def mask_probs_with_excludelist(probs: np.ndarray, class_names: List[str], exclude: set[str], scale: float = 0.0) -> np.ndarray:
    p = probs.copy().astype(float)
    mask = np.array([name in exclude for name in class_names], dtype=bool)
    factor = float(max(0.0, min(scale, 1.0)))
    if p.ndim == 1:
        p[mask] = p[mask] * factor if factor > 0 else 0.0
        s = p.sum()
        return p / s if s > 0 else p
    p[:, mask] = p[:, mask] * factor if factor > 0 else 0.0
    sums = p.sum(axis=1, keepdims=True)
    nz = sums > 0
    p[nz] /= sums[nz]
    return p

def aggregate_probs(list_of_probs: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(list_of_probs) == 0:
        return None
    eps = 1e-9
    logps = [np.log(np.clip(p, eps, 1.0)) for p in list_of_probs]
    mean_logp = np.mean(logps, axis=0)
    exps = np.exp(mean_logp - mean_logp.max())
    return exps / exps.sum()

def apply_thresholds_multi(agg_probs: np.ndarray, thresholds: Dict[str, float], class_names: List[str], top_k_cap: Optional[int] = None) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    g = thresholds.get("__global__", 0.5)
    for i, name in enumerate(class_names):
        thr = thresholds.get(name, g)
        if agg_probs[i] >= thr:
            out.append((name, float(agg_probs[i])))
    out.sort(key=lambda t: t[1], reverse=True)
    if top_k_cap is not None and len(out) > top_k_cap:
        out = out[:top_k_cap]
    return out

# ===== Inference paths (EXPECT bundle.model & bundle.device & stats) =====
@torch.inference_mode()
def predict_windows(y: np.ndarray, sr: int, bundle,
                    stride_frames: int = TARGET_FRAMES // 2) -> List[np.ndarray]:
    mfcc = mfcc_13(y, sr)
    mean, std = getattr(bundle, "mfcc_mean", None), getattr(bundle, "mfcc_std", None)
    mfcc = apply_mfcc_stats(mfcc, mean, std)

    T = mfcc.shape[1]
    out: List[np.ndarray] = []
    if T <= TARGET_FRAMES:
        X = pad_or_crop_frames(mfcc, TARGET_FRAMES)
        p = logits_to_probs(bundle.model(features_to_tensor(X).to(bundle.device)).squeeze(0))
        out.append(p)
        return out

    for start in range(0, T - TARGET_FRAMES + 1, max(1, stride_frames)):
        X = mfcc[:, start:start+TARGET_FRAMES]
        p = logits_to_probs(bundle.model(features_to_tensor(X).to(bundle.device)).squeeze(0))
        out.append(p)
    return out

@torch.inference_mode()
def predict_calls(y: np.ndarray, sr: int, bundle,
                  spans: List[Tuple[int,int]], hop: int = HOP, pad_frames: int = 4) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    mean, std = getattr(bundle, "mfcc_mean", None), getattr(bundle, "mfcc_std", None)
    for (l, r) in spans:
        l2 = max(0, l - pad_frames)
        r2 = r + pad_frames
        t0 = int(l2 * hop)
        t1 = int(r2 * hop)
        chunk = y[t0:t1]
        if len(chunk) < hop * 2:
            continue
        X = mfcc_13(chunk, sr)
        X = apply_mfcc_stats(X, mean, std)
        X = pad_or_crop_frames(X, TARGET_FRAMES)
        p = logits_to_probs(bundle.model(features_to_tensor(X).to(bundle.device)).squeeze(0))
        out.append(p)
    return out
