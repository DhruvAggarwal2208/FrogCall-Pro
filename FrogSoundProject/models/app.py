# app.py — FrogCall Pro (Web)
# Minimal, professional UI. Web-safe (no auto-open, no local shutdown button).
# Assumes you have: infer_utils.py, frog_cnn_best.pt, class_names.json, mfcc_stats.json.
# Default regional whitelist: Oakville, Ontario, Canada.

import io
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---- Web mode flag (hides local-only controls) ----
WEB_MODE = os.environ.get("FROGCALL_MODE", "web")  # default to "web" on hosts

# ---- Import your inference helpers ----
# Expected functions/objects from infer_utils:
#   AUDIO: dict with SR, HOP, N_FFT, N_MFCC, TARGET_FRAMES
#   load_model(model_path, class_names_path, mfcc_stats_path) -> (model, meta)
#   load_wav(file_or_path, sr) -> np.ndarray
#   segment_calls(y, sr, **kwargs) -> list[(start_s, end_s)]
#   windows_fixed(y, sr, window_s, hop_s) -> list[(start_s, end_s)]
#   wav_to_mfcc(y, sr, n_mfcc, n_fft, hop_length, target_frames) -> np.ndarray shape (n_mfcc, T)
#   predict_mfcc_batch(model, mfcc_batch) -> np.ndarray [N, C] softmax probs
#   aggregate_predictions(call_probs, method="logprob_max") -> Dict[class_idx, score]
#   pretty_species_name(scientific: str) -> str
#   load_class_names(path) -> list[str]
#   filter_species(class_names, include=None, exclude=None) -> (idx_keep, class_names_kept)
#
# If your function names differ slightly, adjust calls below to match.
try:
    from infer_utils import (
        AUDIO,
        load_model,
        load_wav,
        segment_calls,
        windows_fixed,
        wav_to_mfcc,
        predict_mfcc_batch,
        aggregate_predictions,
        pretty_species_name,
        load_class_names,
        filter_species,
    )
except Exception as e:
    st.error("Failed to import functions from infer_utils.py. Make sure it exists next to app.py.")
    st.exception(e)
    st.stop()

# ---- Defaults / Constants ----
DEFAULT_MODEL_PATH = os.environ.get("FROGCALL_MODEL", "frog_cnn_best.pt")
DEFAULT_CLASS_NAMES = os.environ.get("FROGCALL_CLASSES", "class_names.json")
DEFAULT_MFCC_STATS = os.environ.get("FROGCALL_MFCC_STATS", "mfcc_stats.json")

# Oakville, Ontario default whitelist (scientific)
OAKVILLE_WHITELIST = [
    "anaxyrus_americanus",      # American Toad
    "hyla_versicolor",          # Gray Treefrog
    "lithobates_clamitans",     # Green Frog
    "lithobates_catesbeianus",  # American Bullfrog
    "pseudacris_crucifer",      # Spring Peeper
]

# ---- Cached loaders ----
@st.cache_resource
def _load_assets(model_path: str, class_names_path: str, mfcc_stats_path: str):
    # load_model should return (model, meta) or at least model; we’ll carry paths in meta if needed
    model, meta = load_model(model_path, class_names_path, mfcc_stats_path)
    class_names = load_class_names(class_names_path)
    return model, class_names, meta

def _pretty_map(names: List[str]) -> Dict[str, str]:
    return {name: pretty_species_name(name) for name in names}

# ---- UI ----
st.set_page_config(page_title="FrogCall Pro — Frog Sound Classifier", page_icon="🐸", layout="wide")

st.title("🐸 FrogCall Pro — Frog Sound Classifier")
st.caption("Website deployment — upload WAV files to classify frog species. CPU-only, web-safe.")

# Sidebar: model + species controls
with st.sidebar:
    st.header("Settings")

    model_path = st.text_input("Model checkpoint (.pt)", value=DEFAULT_MODEL_PATH)
    class_names_path = st.text_input("Class names (JSON)", value=DEFAULT_CLASS_NAMES)
    mfcc_stats_path = st.text_input("MFCC stats (JSON)", value=DEFAULT_MFCC_STATS)

    st.markdown("**Species whitelist preset**")
    use_oakville = st.checkbox("Use Oakville, Ontario preset", value=True, help="Limits predictions to common local species.")
    include_text = st.text_area("Include list (scientific, one per line)", value="\n".join(OAKVILLE_WHITELIST) if use_oakville else "")
    exclude_text = st.text_area("Exclude list (scientific, one per line)", value="", help="Optional: exclude species even if present in include list.")

    # Multi-species vs single-best
    multi_species = st.toggle("Multi-species mode", value=True, help="Return all species above threshold instead of only the top one.")

    # Thresholding
    st.subheader("Thresholds")
    global_thresh = st.slider("Global threshold (prob)", 0.0, 1.0, 0.35, 0.01)
    per_class_json = st.file_uploader("Optional per-class thresholds (.json: {species: threshold})", type=["json"])

    st.subheader("Segmentation")
    seg_mode = st.radio(
        "Method",
        ["Call-activity (spectral-flux)", "Fixed windows"],
        index=0,
        help="Call-activity segments calls; fixed windows is simpler but noisier."
    )
    window_s = st.number_input("Fixed window size (s)", min_value=0.2, max_value=10.0, value=2.0, step=0.1, disabled=(seg_mode != "Fixed windows"))
    hop_s    = st.number_input("Fixed window hop (s)",  min_value=0.1, max_value=10.0, value=0.5, step=0.1, disabled=(seg_mode != "Fixed windows"))

    st.markdown("---")
    st.caption("Tip: For large models, deploy with Git LFS or download from object storage in @st.cache_resource.")

# Main: upload & run
uploaded = st.file_uploader("Upload WAV file(s)", type=["wav"], accept_multiple_files=True)

# Load model on demand (only when we have files)
model = class_names = meta = None
if uploaded:
    with st.spinner("Loading model & assets…"):
        model, class_names, meta = _load_assets(model_path, class_names_path, mfcc_stats_path)
    name_map = _pretty_map(class_names)

    # Include/exclude filters
    include = [x.strip() for x in include_text.splitlines() if x.strip()] if include_text else None
    exclude = [x.strip() for x in exclude_text.splitlines() if x.strip()] if exclude_text else None

    keep_idx, kept_names = filter_species(class_names, include=include, exclude=exclude)
    kept_name_map = {n: name_map[n] for n in kept_names}

    # Per-class thresholds
    per_class = {}
    if per_class_json is not None:
        try:
            per_class = json.load(io.TextIOWrapper(per_class_json, encoding="utf-8"))
        except Exception as e:
            st.warning(f"Could not parse thresholds JSON: {e}")

    # Results accumulator
    all_rows = []

    # Process each file
    for f in uploaded:
        st.markdown(f"### File: `{f.name}`")
        try:
            y = load_wav(f, sr=AUDIO["SR"])
            duration = len(y) / float(AUDIO["SR"])
        except Exception as e:
            st.error(f"Failed to read `{f.name}`: {e}")
            continue

        # Segment into call snippets or fixed windows
        if seg_mode.startswith("Call"):
            segments = segment_calls(y, AUDIO["SR"])  # list of (start_s, end_s)
        else:
            segments = windows_fixed(y, AUDIO["SR"], window_s=window_s, hop_s=hop_s)

        if not segments:
            st.info("No call-like segments detected. Try Fixed windows or lower thresholds in your segmentation settings.")
            continue

        # Extract MFCC for each segment
        mfcc_list = []
        for (s, e) in segments:
            s_i = max(0, int(s * AUDIO["SR"]))
            e_i = min(len(y), int(e * AUDIO["SR"]))
            seg_y = y[s_i:e_i]
            mfcc = wav_to_mfcc(
                seg_y,
                AUDIO["SR"],
                n_mfcc=AUDIO["N_MFCC"],
                n_fft=AUDIO["N_FFT"],
                hop_length=AUDIO["HOP"],
                target_frames=AUDIO["TARGET_FRAMES"],
            )
            mfcc_list.append(mfcc)

        if not mfcc_list:
            st.info("Could not compute MFCCs for any segment.")
            continue

        mfcc_batch = np.stack(mfcc_list, axis=0)  # [N, n_mfcc, T]
        probs = predict_mfcc_batch(model, mfcc_batch)  # [N, C] with softmax

        # Keep only filtered species columns
        probs_kept = probs[:, keep_idx]

        # Aggregate over segments
        agg = aggregate_predictions(probs_kept, method="logprob_max")  # Dict[idx_in_kept -> score]
        # Convert to list of (scientific_name, pretty_name, score)
        scored = []
        for i_kept, score in agg.items():
            sci = kept_names[i_kept]
            thr = per_class.get(sci, global_thresh)
            if score >= thr:
                scored.append((sci, kept_name_map[sci], float(score), float(thr)))

        # Sort by score desc
        scored.sort(key=lambda x: x[2], reverse=True)

        # Multi-species or single-best
        if not multi_species and scored:
            scored = [scored[0]]

        # Show per-file table
        if scored:
            df = pd.DataFrame(scored, columns=["species_scientific", "species_pretty", "score", "threshold"])
            st.dataframe(df, use_container_width=True)
            # Accumulate for CSV
            for row in scored:
                all_rows.append({
                    "file": f.name,
                    "species_scientific": row[0],
                    "species_pretty": row[1],
                    "score": row[2],
                    "threshold": row[3],
                    "duration_s": duration,
                    "segments": len(segments),
                    "segmentation": seg_mode,
                })
        else:
            st.info("No species met the threshold(s) for this file.")

    # Combined CSV download
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", data=csv_bytes, file_name="frogcall_results.csv", mime="text/csv")
