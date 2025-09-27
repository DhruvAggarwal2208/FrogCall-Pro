# FrogSoundProject/src/train_from_loader.py
import os
import json
import time
import math
import random
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------------------
# Paths / Defaults
# ----------------------------
PROJECT_ROOT = "FrogSoundProject"
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
WHITELIST_PATH = os.path.join(PROJECT_ROOT, "data", "species_whitelist.txt")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_classes(use_whitelist: bool = True) -> List[str]:
    """Return sorted list of class names (subfolders in processed)."""
    if use_whitelist and os.path.exists(WHITELIST_PATH):
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
        # Only keep those that actually exist
        classes = [c for c in classes if os.path.isdir(os.path.join(PROC_DIR, c))]
        if classes:
            return sorted(classes)
    # Fallback: all dirs
    classes = [
        d for d in os.listdir(PROC_DIR)
        if os.path.isdir(os.path.join(PROC_DIR, d))
    ]
    return sorted(classes)

def load_class_index_map(classes: List[str]) -> dict:
    return {c: i for i, c in enumerate(classes)}

def list_npy_files_for_class(cls: str) -> List[str]:
    cdir = os.path.join(PROC_DIR, cls)
    return sorted([os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".npy")])

def pad_or_crop_frames(arr: np.ndarray, target_frames: int, train: bool) -> np.ndarray:
    """
    arr: (13, T)
    Returns (13, target_frames)
    """
    n_mels, T = arr.shape
    if T == target_frames:
        return arr
    if T < target_frames:
        # pad at end
        pad = target_frames - T
        return np.pad(arr, ((0,0),(0,pad)), mode="constant")
    # T > target_frames
    if train:
        start = random.randint(0, T - target_frames)
    else:
        start = max(0, (T - target_frames)//2)
    return arr[:, start:start+target_frames]

def augment_mfcc(arr: np.ndarray) -> np.ndarray:
    """
    Simple MFCC-space augmentation: random gain, gaussian noise, circular time shift
    arr: (13, T)
    """
    out = arr.copy()
    # random gain in MFCC space (small)
    if random.random() < 0.5:
        g = 1.0 + random.uniform(-0.15, 0.15)
        out *= g
    # gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.02, size=out.shape).astype(out.dtype)
        out += noise
    # time shift (circular)
    if random.random() < 0.5 and out.shape[1] > 1:
        shift = random.randint(-out.shape[1]//10, out.shape[1]//10)
        out = np.roll(out, shift, axis=1)
    return out

# ----------------------------
# Dataset
# ----------------------------
class MFCCDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        target_frames: int = 100,
        train: bool = True,
        augment_p: float = 0.0,
    ):
        """
        items: list of (path, label_index)
        """
        self.items = items
        self.target_frames = target_frames
        self.train = train
        self.augment_p = augment_p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        a = np.load(p)  # expect (13, T)
        if a.ndim != 2 or a.shape[0] != 13:
            # try to coerce common mistakes: (T, 13)
            if a.ndim == 2 and a.shape[1] == 13:
                a = a.T
            else:
                raise ValueError(f"Bad MFCC shape {a.shape} for {p}, expected (13, T)")
        a = pad_or_crop_frames(a, self.target_frames, train=self.train)
        if self.train and self.augment_p > 0 and random.random() < self.augment_p:
            a = augment_mfcc(a)
        # to tensor (1, 13, T)
        x = torch.from_numpy(a).float().unsqueeze(0)
        return x, y

# ----------------------------
# Model
# ----------------------------
class SmallCNN(nn.Module):
    """
    A compact 2D CNN for (1, 13, frames) inputs.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((13, 1)),  # collapse time
            nn.Flatten(),
            nn.Linear(64 * 13 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)   # (B, C, 13, T')
        x = self.classifier(x) # (B, n_classes)
        return x

# ----------------------------
# Data prep
# ----------------------------
def build_items(classes: List[str], min_per_class: int) -> List[Tuple[str,int]]:
    class_to_idx = load_class_index_map(classes)
    items = []
    kept_classes = []
    for c in classes:
        files = list_npy_files_for_class(c)
        if len(files) >= min_per_class:
            kept_classes.append(c)
            y = class_to_idx[c]
            for p in files:
                items.append((p, y))
    return items, kept_classes

def stratified_split(items: List[Tuple[str,int]], train_split: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for p, y in items:
        by_class[y].append((p, y))
    train, val = [], []
    for y, lst in by_class.items():
        rng.shuffle(lst)
        n_train = max(1, int(len(lst) * train_split))
        train.extend(lst[:n_train])
        val.extend(lst[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def make_balanced_sampler(train_items: List[Tuple[str,int]], n_classes: int):
    counts = Counter([y for _, y in train_items])
    weights = [1.0 / counts[y] for _, y in train_items]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    return sampler, counts

# ----------------------------
# Metrics
# ----------------------------
def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean()) if len(true) else 0.0

def per_class_report(y_true: List[int], y_pred: List[int], class_names: List[str], top_k:int=10) -> str:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.arange(len(class_names))
    report_lines = ["Per-class (top by support): name | n | precision | recall | f1"]
    supports = {c: int((y_true == c).sum()) for c in classes}
    order = sorted(classes, key=lambda c: supports[c], reverse=True)[:top_k]
    for c in order:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        report_lines.append(f"  {class_names[c]:<28} | {supports[c]:>4} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f}")
    return "\n".join(report_lines)

# ----------------------------
# Train / Eval
# ----------------------------
def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n = 0
    all_y = []
    all_p = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_p.append(preds)
        all_y.append(yb.detach().cpu().numpy())
    all_p = np.concatenate(all_p) if all_p else np.array([])
    all_y = np.concatenate(all_y) if all_y else np.array([])
    acc = accuracy(all_p, all_y)
    return (total_loss / max(1, n)), acc, all_y, all_p

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a CNN on processed MFCC segments.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--balanced", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=100, help="Target number of MFCC frames")
    parser.add_argument("--augment-p", type=float, default=0.2, help="Prob. of applying MFCC augmentations (train only)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--min-per-class", type=int, default=1, help="Drop classes with fewer than this many files")
    args = parser.parse_args()

    seed_everything(args.seed)

    classes = list_classes(use_whitelist=True)
    items, kept_classes = build_items(classes, min_per_class=args.min_per_class)
    if not kept_classes:
        print("No classes with enough items. Check your processed data / whitelist.")
        return
    classes = kept_classes
    class_to_idx = load_class_index_map(classes)
    n_classes = len(classes)
    print(f"Found {n_classes} classes.")

    # Split
    train_items, val_items = stratified_split(items, train_split=args.train_split, seed=args.seed)
    print(f"Train items: {len(train_items)} | Val items: {len(val_items)}")

    # Datasets
    ds_train = MFCCDataset(train_items, target_frames=args.frames, train=True, augment_p=args.augment_p)
    ds_val   = MFCCDataset(val_items,   target_frames=args.frames, train=False, augment_p=0.0)

    # Sampler / Loaders
    if args.balanced:
        sampler, counts = make_balanced_sampler(train_items, n_classes)
        print("Using class-balanced sampler. Train class counts (subset):")
        # Show top 10 largest classes in train split
        inv_map = {v:k for k,v in class_to_idx.items()}
        for cls_idx, cnt in Counter([y for _, y in train_items]).most_common(10):
            print(f"  {inv_map[cls_idx]}: {cnt}")
        train_loader = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(n_classes).to(device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # Train
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_path = os.path.join(MODELS_DIR, "frog_cnn_best.pt")
    label_map_path = os.path.join(MODELS_DIR, "label_map.json")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs:02d}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}")

        # Save best
        improved = (val_acc > best_val_acc) or (math.isclose(val_acc, best_val_acc) and val_loss < best_val_loss)
        if improved:
            best_val_acc, best_val_loss = val_acc, val_loss
            torch.save({"model_state": model.state_dict(),
                        "n_classes": n_classes,
                        "classes": classes,
                        "frames": args.frames}, best_path)
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump({"index_to_label": {i: c for i, c in enumerate(classes)}}, f, indent=2)

    # Final evaluation report on last epoch
    print("\nBest checkpoint metrics:")
    print(f"  Val acc: {best_val_acc:.3f} | Val loss: {best_val_loss:.4f}")

    # Per-class report (last epoch preds)
    print()
    print(per_class_report(y_true.tolist(), y_pred.tolist(), classes, top_k=min(20, len(classes))))

    print(f"\n✅ Saved model: {best_path}")
    print(f"✅ Saved label map: {label_map_path}")
    print(f"⏱️  Done in {((time.time()-t0)/60):.1f} min")


if __name__ == "__main__":
    main()
