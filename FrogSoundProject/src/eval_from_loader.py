# FrogSoundProject/src/eval_from_loader.py
import os, json, argparse, random, numpy as np
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt # type: ignore

PROC_DIR = os.path.join("FrogSoundProject", "data", "processed")
MODEL_DIR = os.path.join("FrogSoundProject", "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "frog_cnn_best.pt")
DEFAULT_LABELMAP = os.path.join(MODEL_DIR, "label_map.json")

# ---------- Dataset ----------
class MFCCDataset(Dataset):
    def __init__(self, items, target_frames=100, add_channel_dim=True):
        self.items = items
        self.T = target_frames
        self.add_channel = add_channel_dim

    def __len__(self):
        return len(self.items)

    def _pad_or_crop(self, a):
        c, t = a.shape
        if t < self.T:
            pad = np.zeros((c, self.T - t), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=1)
        elif t > self.T:
            a = a[:, :self.T]
        return a

    def __getitem__(self, idx):
        path, y = self.items[idx]
        a = np.load(path)
        if a.ndim == 3:
            a = a.squeeze(-1)
        a = a.astype(np.float32)
        a = self._pad_or_crop(a)
        if self.add_channel:
            a = a[None, ...]  # (1, 13, T)
        return torch.from_numpy(a), y

# ---------- Model (features/classifier names must match checkpoint) ----------
class SmallCNN(nn.Module):
    def __init__(self, n_classes, pool_time_bins=1, in_ch=1, classifier_in=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),     # 0
            nn.BatchNorm2d(16),                     # 1
            nn.ReLU(),                              # 2
            nn.MaxPool2d((1, 2)),                   # 3
            nn.Conv2d(16, 32, 3, padding=1),        # 4
            nn.BatchNorm2d(32),                     # 5
            nn.ReLU(),                              # 6
            nn.MaxPool2d((1, 2)),                   # 7
            nn.Conv2d(32, 64, 3, padding=1),        # 8
            nn.BatchNorm2d(64),                     # 9
            nn.ReLU(),                              # 10
            # We’ll set the time bins dynamically; freq is fixed at 13
            nn.AdaptiveMaxPool2d((13, pool_time_bins)),  # 11
        )
        in_feats = classifier_in if classifier_in is not None else 64 * 13 * pool_time_bins
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # 0
            nn.Dropout(0.0),                        # 1  (no weights; matches your checkpoint indexing)
            nn.Linear(in_feats, 256),               # 2  (has weights)
            nn.ReLU(),                              # 3
            nn.Dropout(0.3),                        # 4  (no weights)
            nn.Linear(256, n_classes),              # 5  (has weights)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------- Helpers ----------
def normalize_label_map(obj):
    if isinstance(obj, dict):
        if all(isinstance(v, int) for v in obj.values()):
            n = max(obj.values()) + 1
            classes = [None]*n
            for name, idx in obj.items():
                classes[idx] = name
            return classes
        if all(str(k).isdigit() for k in obj.keys()):
            items = {int(k): v for k, v in obj.items()}
            n = max(items.keys()) + 1
            classes = [None]*n
            for idx, name in items.items():
                classes[idx] = name
            return classes
        return list(obj.keys())
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(x, str) for x in obj):
            return list(obj)
        if all(isinstance(x, dict) and ("index" in x) for x in obj):
            n = max(int(x["index"]) for x in obj) + 1
            classes = [None]*n
            for x in obj:
                classes[int(x["index"])] = x.get("name", str(x.get("label", "?")))
            return classes
        return [x.get("name", str(x.get("label", f"class_{i}"))) if isinstance(x, dict) else str(x)
                for i, x in enumerate(obj)]
    return []

def load_classes(label_map_path, ckpt):
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, dict)):
        cls_from_ckpt = normalize_label_map(ckpt["classes"])
        cls_from_ckpt = [c for c in cls_from_ckpt if c is not None]
        if len(cls_from_ckpt) > 0:
            print(f"Loaded {len(cls_from_ckpt)} classes from checkpoint.")
            return cls_from_ckpt
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    classes = normalize_label_map(raw)
    classes = [c for c in classes if c is not None]
    print(f"Loaded {len(classes)} classes from label_map.json")
    return classes

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if any(isinstance(k, str) and (k.startswith("features.") or k.startswith("classifier.")) for k in ckpt.keys()):
            return ckpt
    return ckpt

def build_items(proc_dir, classes):
    items = []
    for i, cls in enumerate(classes):
        for fp in glob(os.path.join(proc_dir, cls, "*.npy")):
            items.append((fp, i))
    return items

def split_items(items, train_split=0.78, seed=1337):
    random.Random(seed).shuffle(items)
    n_train = int(len(items) * train_split)
    return items[:n_train], items[n_train:]

def plot_confusion(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix (Validation)')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Evaluate frog model on validation split.")
    ap.add_argument("--processed", default=PROC_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label-map", default=DEFAULT_LABELMAP)
    ap.add_argument("--train-split", type=float, default=0.78)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--frames", type=int, default=None, help="Override frames; default uses checkpoint if present, else 100")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # Load checkpoint & classes
    ckpt = torch.load(args.model, map_location="cpu")
    classes = load_classes(args.label_map, ckpt)

    # Frames
    target_frames = args.frames if args.frames is not None else int(ckpt.get("frames", 100))
    print(f"Using target frames: {target_frames}")

    # Build splits
    all_items = build_items(args.processed, classes)
    train_items, val_items = split_items(all_items, train_split=args.train_split, seed=args.seed)
    print(f"Classes: {len(classes)} | Train: {len(train_items)} | Val: {len(val_items)}")
    if len(val_items) == 0:
        print("No validation items found. Aborting.")
        return

    val_loader = DataLoader(
        MFCCDataset(val_items, target_frames=target_frames, add_channel_dim=True),
        batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False
    )

    # Figure out the FC input size expected by the checkpoint
    state_dict = extract_state_dict(ckpt)
    if "classifier.2.weight" not in state_dict:
        raise RuntimeError("Checkpoint does not contain 'classifier.2.weight'; cannot infer FC size.")
    fc_in_from_ckpt = state_dict["classifier.2.weight"].shape[1]   # e.g., 832
    # infer pool_time_bins given conv out channels (64) and freq bins 13
    if fc_in_from_ckpt % (64*13) != 0:
        raise RuntimeError(f"Unexpected FC in_features {fc_in_from_ckpt}; not divisible by 64*13.")
    pool_time_bins = fc_in_from_ckpt // (64*13)

    # Build model to MATCH checkpoint geometry exactly
    model = SmallCNN(n_classes=len(classes), pool_time_bins=pool_time_bins, classifier_in=fc_in_from_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ load_state_dict warnings — missing: {missing}, unexpected: {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Eval
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().tolist())

    # Metrics
    labels = list(range(len(classes)))
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    print(f"\nOverall:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1 (micro): {f1_micro:.3f}")
    print(f"  F1 (macro): {f1_macro:.3f}")

    print("\nPer-class: name | n | precision | recall | f1")
    order = np.argsort(-support)
    for idx in order[:20]:
        print(f"  {classes[idx]:<28} | {int(support[idx]):4d} | {prec[idx]:6.2f} | {rec[idx]:6.2f} | {f1[idx]:6.2f}")

    # Confusion matrix & outputs
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    os.makedirs(MODEL_DIR, exist_ok=True)
    cm_png = os.path.join(MODEL_DIR, "confusion_matrix_val.png")
    plot_confusion(cm, classes, cm_png)
    print(f"\n📈 Confusion matrix saved to: {cm_png}")

    out_csv = os.path.join(MODEL_DIR, "val_predictions.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("true_label,true_name,pred_label,pred_name\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{t},{classes[t]},{p},{classes[p]}\n")
    print(f"📝 Predictions CSV: {out_csv}")

    report_txt = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0))
    print(f"📄 Detailed report: {report_txt}")

if __name__ == "__main__":
    main()
