# export_torchscript.py (robust)
from __future__ import annotations
import argparse, importlib, json, os, sys, types
from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
from collections import OrderedDict

# ---------- utilities ----------
def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "classes" in obj:
        return list(obj["classes"])
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError("class_names.json must be a list or a dict with key 'classes'")

def try_import_build(num_classes: int):
    """
    Try to locate your real model builder/class inside your repo.
    If you know the exact file/class, replace this with a direct import.
    """
    # Make sure repo root is importable
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "FrogSoundProject"))
    sys.path.insert(0, os.path.join(os.getcwd(), "FrogSoundProject", "src"))

    CANDIDATE_MODULES = [
        "FrogSoundProject.src.model",
        "FrogSoundProject.src.models",
        "FrogSoundProject.src.networks",
        "FrogSoundProject.src.train_from_loader",
        "model", "models", "net", "network",
    ]
    CANDIDATE_BUILDERS = ["build_model", "make_model", "create_model", "get_model"]
    CANDIDATE_CLASSES  = ["Net", "SmallCNN", "AudioCNN", "Model", "CNN"]

    for mod in CANDIDATE_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for fn in CANDIDATE_BUILDERS:
            if hasattr(m, fn):
                try:
                    return getattr(m, fn)(num_classes=num_classes)  # type: ignore
                except TypeError:
                    try:
                        return getattr(m, fn)(num_classes)  # positional
                    except Exception:
                        pass
        for cls in CANDIDATE_CLASSES:
            if hasattr(m, cls):
                try:
                    return getattr(m, cls)(num_classes=num_classes)  # type: ignore
                except TypeError:
                    try:
                        return getattr(m, cls)(num_classes)  # positional
                    except Exception:
                        pass
    return None

class FallbackSmallCNN(nn.Module):
    """Only for fallback if we cannot locate your true model."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):  # x: [B,1,13,100]
        return self.net(x)

# ---------- checkpoint readers ----------
def _is_state_dict_like(x: Any) -> bool:
    if isinstance(x, (dict, OrderedDict)) and len(x) > 0:
        # allow non-tensor values to pass; we’ll filter below
        return True
    return False

def _extract_state_dict(obj: Any) -> Optional[OrderedDict]:
    """
    Try many common layouts to find an OrderedDict of parameter tensors.
    """
    candidates = []
    if isinstance(obj, nn.Module):
        return obj.state_dict()  # full module; caller will prefer module path
    if isinstance(obj, (OrderedDict, dict)):
        # direct state_dict?
        if all(torch.is_tensor(v) for v in obj.values()):
            return OrderedDict(obj)
        # lightning / custom wrappers
        for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, nn.Module):
                    return v.state_dict()
                if isinstance(v, (OrderedDict, dict)) and len(v) > 0:
                    if all(torch.is_tensor(t) for t in v.values()):
                        return OrderedDict(v)
        # nested dicts; pick the deepest dict-of-tensors
        def walk(d):
            if isinstance(d, (OrderedDict, dict)):
                if len(d) and all(torch.is_tensor(v) for v in d.values()):
                    candidates.append(d)
                for v in d.values():
                    walk(v)
        walk(obj)
        if candidates:
            # choose the largest number of tensors as best guess
            best = max(candidates, key=lambda d: sum(1 for _ in d.items()))
            return OrderedDict(best)
    return None

def read_checkpoint_any(path: str):
    # 1) TorchScript?
    try:
        ts = torch.jit.load(path, map_location="cpu")
        ts.eval()
        return {"type": "torchscript", "module": ts}
    except Exception:
        pass
    # 2) torch.load anything
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, nn.Module):
        obj.eval()
        return {"type": "module", "module": obj}
    if _is_state_dict_like(obj):
        sd = _extract_state_dict(obj)
        if sd is not None:
            return {"type": "state_dict", "state_dict": sd, "root_keys": list(obj.keys()) if isinstance(obj, dict) else []}
    # unknown
    return {"type": "unknown", "python_type": type(obj).__name__, "repr": repr(obj)[:300]}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt/.pth)")
    ap.add_argument("--classes", default="class_names.json")
    ap.add_argument("--out", default="frog_cnn_best_script.pt")
    args = ap.parse_args()

    classes = load_class_names(args.classes)
    num_classes = len(classes)
    print(f"[i] num_classes = {num_classes}")

    ck = read_checkpoint_any(args.ckpt)
    if ck["type"] == "torchscript":
        print("[i] Input is already TorchScript → copying to output.")
        torch.jit.save(ck["module"], args.out)
        print(f"[✓] Saved TorchScript to: {args.out}")
        return 0

    if ck["type"] == "module":
        model = ck["module"]
        print("[i] Loaded full nn.Module from checkpoint.")
    elif ck["type"] == "state_dict":
        print("[i] Found state_dict in checkpoint.")
        model = try_import_build(num_classes)
        if model is None:
            print("[!] Could not auto-locate your model in FrogSoundProject/src — using FallbackSmallCNN.")
            model = FallbackSmallCNN(num_classes)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        print(f"[i] load_state_dict(strict=False): Missing={len(missing)} Unexpected={len(unexpected)}")
    else:
        print("[!] Unsupported checkpoint content:")
        print(f"    python_type={ck.get('python_type')} repr={ck.get('repr')}")
        print("[!] This file is not a PyTorch checkpoint I recognize. "
              "If it’s from Keras/TF, use the Keras model instead, or export a PyTorch state_dict.")
        return 2

    model.eval()
    example = torch.zeros(1,1,13,100, dtype=torch.float32)
    ts = torch.jit.trace(model, example)
    torch.jit.save(ts, args.out)
    print(f"[✓] Saved TorchScript to: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())