# FrogSoundProject/src/create_whitelist_topk.py
import os, argparse

PROC = os.path.join("FrogSoundProject","data","processed")

DEFAULT_EXCLUDES = {
    "INCT17","misc","am_edited","pm_edited","mod","processed","prepared"
}

def main():
    ap = argparse.ArgumentParser(description="Create whitelist of species from data/processed")
    ap.add_argument("--k", type=int, default=180, help="How many classes to keep")
    ap.add_argument("--min-per-class", type=int, default=8, help="Minimum .npy files required per class")
    ap.add_argument("--out", default=os.path.join("FrogSoundProject","data","species_whitelist.txt"))
    ap.add_argument("--allow-junk", action="store_true",
                    help="If set, do NOT auto-exclude legacy/junk buckets (INCT17, misc, etc.)")
    ap.add_argument("--extra-exclude", nargs="*", default=[],
                    help="Additional class names to exclude")
    args = ap.parse_args()

    excludes = set(args.extra_exclude)
    if not args.allow_junk:
        excludes |= DEFAULT_EXCLUDES

    print(f"📂 Scanning: {PROC}")
    if not os.path.isdir(PROC):
        print("❌ processed dir not found.")
        return

    rows = []          # (name, count)
    seen = kept = dropped = 0

    for d in sorted(os.listdir(PROC)):
        full = os.path.join(PROC, d)
        if not os.path.isdir(full):
            continue
        seen += 1

        if d in excludes:
            dropped += 1
            continue

        npys = [f for f in os.listdir(full) if f.endswith(".npy")]
        n = len(npys)
        if n >= args.min_per_class:
            rows.append((d, n))
            kept += 1

    rows.sort(key=lambda x: x[1], reverse=True)
    picked = [name for name,_ in rows[:args.k]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for name in picked:
            f.write(name + "\n")

    print(f"\n🔎 Classes seen: {seen}")
    print(f"🚫 Excluded by name: {dropped}  (use --allow-junk to include them)")
    print(f"✅ Eligible (≥ {args.min_per_class} files): {kept}")
    print(f"🎯 Selected top-K = {len(picked)} -> {args.out}")

    top_show = min(10, len(rows))
    if top_show:
        print("\nTop classes by count:")
        for name, n in rows[:top_show]:
            print(f"  {name}: {n}")
    else:
        print("\n(No classes met the min-per-class threshold. Try lowering --min-per-class.)")

if __name__ == "__main__":
    main()
