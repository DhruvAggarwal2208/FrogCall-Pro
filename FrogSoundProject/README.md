# 🐸 FrogCall Pro — Frog Sound Classifier (Web)

**FrogCall Pro** is a Streamlit web app that classifies frog species from WAV audio.
It uses a CNN trained on MFCC features and supports call-activity segmentation, multi-species detection,
per-class thresholds, and CSV export.

> **Deployment:** Website (Streamlit Cloud or Render). No local launchers or EXEs required.

---

## Features

- Upload one or many WAVs; get predictions per file
- Call-activity segmentation (spectral-flux) or fixed windows
- Multi-species mode (return all species above thresholds) or single-best
- Thresholds: global slider + optional per-class JSON
- Regional whitelist (default preset: *Oakville, Ontario, Canada*)
- Download results as CSV (file, species, score, threshold, segments, etc.)
- CPU-only compatible on common hosts (Streamlit Cloud, Render)


## In-App Attribution Footer

The app displays a small footer reminding users that audio on iNaturalist is licensed per record:

> Audio source: iNaturalist (per-record licenses; attribute observers & iNaturalist as required).

*(If you fork or white-label this app, keep an equivalent attribution notice.)*



## FAQ

**Does the app include third-party audio or demo clips?**  
No. The hosted app does not bundle any iNaturalist audio. Users upload their own WAV files. If you distribute any media yourself, you must comply with the license attached to each iNaturalist observation (e.g., CC BY / CC BY-NC / CC0 / All Rights Reserved).

**What license applies to this repo?**  
Code is MIT (no warranty, no liability). Media/metadata from iNaturalist are licensed per record and are not MIT-licensed; follow their terms.
