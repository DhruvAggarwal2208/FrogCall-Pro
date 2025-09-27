# FrogSoundProject/src/download_inat_frogs.py
import os, sys, json, time, math, re, argparse, pathlib, shutil
from typing import Dict, Set, Tuple, List
import requests
from urllib.parse import urlencode

RAW_ROOT = os.path.join("FrogSoundProject", "data", "raw")
OUT_ROOT = os.path.join(RAW_ROOT, "inat_frogs")
MANIFEST = os.path.join(OUT_ROOT, "_manifest.json")

INAT_API = "https://api.inaturalist.org/v1"
ANURA_TAXON_ID = 20978  # Order: Anura (frogs & toads)

DEFAULT_SPECIES_TARGET = 180
DEFAULT_MAX_PER_SPECIES = 10
MIN_SECONDS = 5.0
REQUESTS_TIMEOUT = 30
RATE_DELAY = 0.4  # gentle on API

SAFE_CHARS = re.compile(r"[^a-z0-9_\-]")

def safe_slug(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    return SAFE_CHARS.sub("_", s)

def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)

def load_manifest() -> Dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"species": {}, "downloaded_files": {}}

def save_manifest(m: Dict):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MANIFEST)

def head_duration_seconds(path: str) -> float:
    """
    Fast duration check: prefer mutagen if available; fall back to librosa.
    """
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(path)
        if mf is not None and getattr(mf, "info", None) and mf.info.length:
            return float(mf.info.length)
    except Exception:
        pass
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True, duration=60.0)  # cap load
        return float(librosa.get_duration(y=y, sr=sr))
    except Exception:
        return 0.0

def get_json(session: requests.Session, endpoint: str, params: Dict) -> Dict:
    for attempt in range(5):
        try:
            url = f"{INAT_API}/{endpoint}?{urlencode(params, doseq=True)}"
            r = session.get(url, timeout=REQUESTS_TIMEOUT)
            if r.status_code == 429:
                # rate limited
                time.sleep(2.0 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1.0 + attempt * 1.0)
    return {}

def stream_observations(session: requests.Session, max_pages: int = 400):
    """
    Yields observation JSONs that include sounds for Anura.
    """
    page = 1
    per_page = 200
    while page <= max_pages:
        params = {
            "taxon_id": ANURA_TAXON_ID,
            "sounds": "true",
            "quality_grade": "research,needs_id",
            "order_by": "created_at",
            "order": "desc",
            "page": page,
            "per_page": per_page,
            "locale": "en",
        }
        data = get_json(session, "observations", params)
        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            yield obs
        page += 1
        time.sleep(RATE_DELAY)

def download_file(session: requests.Session, url: str, dest: str) -> bool:
    tmp = dest + ".part"
    try:
        with session.get(url, stream=True, timeout=REQUESTS_TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)
        return True
    except Exception:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download frog recordings from iNaturalist (auto-group by species)."
    )
    parser.add_argument("--species_target", type=int, default=DEFAULT_SPECIES_TARGET,
                        help=f"Aim for this many species (default {DEFAULT_SPECIES_TARGET}).")
    parser.add_argument("--max_per_species", type=int, default=DEFAULT_MAX_PER_SPECIES,
                        help=f"Max clips per species (default {DEFAULT_MAX_PER_SPECIES}).")
    parser.add_argument("--resume", action="store_true", help="Resume from manifest if present.")
    parser.add_argument("--min_seconds", type=float, default=MIN_SECONDS,
                        help=f"Minimum audio duration to keep (default {MIN_SECONDS}s).")
    args = parser.parse_args()

    ensure_dirs()
    manifest = load_manifest() if args.resume else {"species": {}, "downloaded_files": {}}
    species_counts: Dict[str, int] = {k: v.get("count", 0) for k, v in manifest.get("species", {}).items()}

    species_set: Set[str] = set(species_counts.keys())
    session = requests.Session()
    session.headers.update({
        "User-Agent": "FrogSoundProject/1.0 (contact: your_email@example.com)"
    })

    print(f"🎯 Target: {args.species_target} species, ≤{args.max_per_species} clips/species, min {args.min_seconds:.1f}s")
    kept_total = sum(species_counts.values())

    try:
        for obs in stream_observations(session):
            taxon = obs.get("taxon") or {}
            if not taxon or taxon.get("rank") != "species":
                continue
            sci = taxon.get("name") or ""
            if not sci:
                continue
            slug = safe_slug(sci)
            if species_counts.get(slug, 0) >= args.max_per_species:
                continue
            sounds = obs.get("sounds") or []
            if not sounds:
                continue

            # choose a sound file url
            file_url = None
            for s in sounds:
                file_url = s.get("file_url") or s.get("ogg_url") or s.get("mp3_url")
                if file_url:
                    break
            if not file_url:
                continue

            # Prepare dest
            species_dir = os.path.join(OUT_ROOT, slug)
            os.makedirs(species_dir, exist_ok=True)
            obs_id = obs.get("id")
            sound_id = sounds[0].get("id") if sounds else "snd"
            ext = pathlib.Path(file_url.split("?")[0]).suffix
            if not ext:
                ext = ".mp3"
            dest = os.path.join(species_dir, f"{slug}_obs{obs_id}_s{sound_id}{ext}")

            if os.path.exists(dest):
                # Count it (resume safety) if duration ok
                if head_duration_seconds(dest) >= args.min_seconds:
                    species_counts[slug] = species_counts.get(slug, 0) + 1
                    manifest["species"].setdefault(slug, {"count": 0})
                    manifest["species"][slug]["count"] = species_counts[slug]
                    save_manifest(manifest)
                continue

            ok = download_file(session, file_url, dest)
            if not ok:
                continue

            dur = head_duration_seconds(dest)
            if dur < args.min_seconds:
                try: os.remove(dest)
                except Exception: pass
                continue

            # keep
            species_counts[slug] = species_counts.get(slug, 0) + 1
            species_set.add(slug)
            manifest["species"].setdefault(slug, {"count": 0})
            manifest["species"][slug]["count"] = species_counts[slug]
            manifest["downloaded_files"][dest] = {"duration": dur}
            save_manifest(manifest)

            kept_total += 1
            # status
            if kept_total % 10 == 0:
                print(f"✅ {kept_total} clips | {len([s for s,c in species_counts.items() if c>0])} species so far…")

            # stop when we have the *species* target (not total clips)
            if len([s for s, c in species_counts.items() if c >= 1]) >= args.species_target:
                break

            time.sleep(RATE_DELAY)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted, manifest saved.")
    finally:
        have_species = sorted([s for s, c in species_counts.items() if c >= 1])
        print(f"\n📦 Done. Species collected: {len(have_species)} (target {args.species_target})")
        print(f"📁 Raw audio in: {OUT_ROOT}")
        print(f"🧾 Manifest: {MANIFEST}")
        if have_species:
            print("Examples:", ", ".join(have_species[:8]), "…")

if __name__ == "__main__":
    main()
