"""Build a clean training set AND a held-out test set from big public datasets,
so you do not have to collect photos by hand.

Run with the venv active, from the waste-twin folder:
    python prepare_dataset.py "C:\\Users\\adhit\\OneDrive\\Documents\\waste-twin\\data set"
    python prepare_dataset.py SOURCE1 SOURCE2        (several datasets can be merged)
    python prepare_dataset.py SOURCE --per-class 200 --test 20

Each SOURCE must hold one sub-folder per class (nested folders like train/plastic are fine).
It creates:
    dataset_auto/<class>/   up to --per-class photos per class (default 300)
    testset_auto/<class>/   --test photos per class (default 25) that training never sees
Exact duplicate files are removed, damaged files are skipped, and every photo is
resized to at most 512 px, so training is faster. Your source folders are not changed.
"""
import argparse
import hashlib
import os
import random
import sys

from PIL import Image

OUT_TRAIN, OUT_TEST = "dataset_auto", "testset_auto"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# source folder name (lowercase) -> class name used by the app. Folders not listed here are skipped.
ALIASES = {
    "plastic": "plastic", "paper": "paper", "metal": "metal", "glass": "glass", "cardboard": "cardboard",
    "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass",
    "organic waste": "organic", "organic": "organic", "biological": "organic", "food waste": "organic",
    "trash": "other", "other": "other",
    "battery": "e-waste", "batteries": "e-waste", "e-waste": "e-waste", "ewaste": "e-waste",
    "medicine": "medicine",
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("sources", nargs="+")
ap.add_argument("--per-class", type=int, default=300)
ap.add_argument("--test", type=int, default=25)
args = ap.parse_args()

for out in (OUT_TRAIN, OUT_TEST):
    if os.path.exists(out):
        sys.exit(f"'{out}' already exists. Rename or delete it first, so runs are not mixed.")

found, unknown, seen = {}, {}, set()
for src in args.sources:
    if not os.path.isdir(src):
        sys.exit(f"Source folder not found: {src}")
    for root, _, files in os.walk(src):
        imgs = [f for f in files if f.lower().endswith(EXTS)]
        if not imgs:
            continue
        name = os.path.basename(root).lower().strip()
        cls = ALIASES.get(name)
        if cls is None:
            unknown[name] = unknown.get(name, 0) + len(imgs)
            continue
        for f in imgs:
            p = os.path.join(root, f)
            try:
                with open(p, "rb") as fh:
                    digest = hashlib.md5(fh.read()).hexdigest()
            except OSError:
                continue
            if digest not in seen:
                seen.add(digest)
                found.setdefault(cls, []).append(p)


def save(path, folder, name):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((512, 512))
            os.makedirs(folder, exist_ok=True)
            im.save(os.path.join(folder, name), "JPEG", quality=90)
        return True
    except Exception:
        return False


if not found:
    sys.exit("No usable class folders found. Folder names must be like plastic, paper, metal, glass, cardboard, organic waste, trash.")

summary = {}
for cls, paths in sorted(found.items()):
    random.Random(42).shuffle(paths)
    n_test = n_train = 0
    for p in paths:
        if n_test >= args.test and n_train >= args.per_class:
            break
        if n_test < args.test:
            if save(p, os.path.join(OUT_TEST, cls), f"{cls}_{n_test:04d}.jpg"):
                n_test += 1
        elif save(p, os.path.join(OUT_TRAIN, cls), f"{cls}_{n_train:04d}.jpg"):
            n_train += 1
    summary[cls] = (n_train, n_test, len(paths))

print(f"\n{'class':12s} {'train':>6s} {'test':>6s} {'available':>10s}")
for cls, (a, b, c) in summary.items():
    print(f"{cls:12s} {a:6d} {b:6d} {c:10d}")
if unknown:
    print("\nSkipped folders (name not recognised):", ", ".join(f"{k} ({v})" for k, v in list(unknown.items())[:10]))
low = [c for c, (a, _, _) in summary.items() if a < 60]
if low:
    print("\nWARNING: few photos for:", ", ".join(low), "- accuracy will be weaker for these.")
print(f"\nNext:  python train_model.py {OUT_TRAIN}\nThen:  python check_model.py keras_model.h5 {OUT_TEST}")