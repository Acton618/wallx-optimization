import json
import numpy as np
import torch
from safetensors.torch import load_file, save_file

CKPT = "/opt/data/zhuoziying/wall-x-model/wall-oss-0.5/model.safetensors"
NORM = "/opt/data/zhuoziying/dataset/lerobot_v0/libero/norm_stats_libero.json"
OUT  = "/opt/data/zhuoziying/wall-x-model/wall-oss-0.5/model_with_libero.safetensors"

TARGET_DIM = 26
NAME = "libero_all"

sd = load_file(CKPT, device="cpu")
raw = json.load(open(NORM))["norm_stats"]

state = raw["observation.state"]
action = raw["action"]

def to_tensor(x):
    return torch.from_numpy(x.copy()).to(torch.float32)

def pad(vec, fill=0.0):
    arr = np.asarray(vec, dtype=np.float32)
    out = np.full((TARGET_DIM,), fill, dtype=np.float32)
    out[:len(arr)] = arr
    return to_tensor(out)

def pad_std(vec):
    arr = np.asarray(vec, dtype=np.float32)
    out = np.ones((TARGET_DIM,), dtype=np.float32)
    out[:len(arr)] = arr
    return to_tensor(out)

# ---------- propri (state: 8 dim) ----------
for suffix, fn in [
    ("mean", lambda v: pad(v, 0.0)),
    ("std",  pad_std),
    ("q01",  lambda v: pad(v, 0.0)),
    ("q99",  lambda v: pad(v, 0.0)),
]:
    key = f"action_preprocessor.normalizer_propri.{suffix}.{NAME}"
    sd[key] = fn(state[suffix])
    print(f"  + {key}  shape={tuple(sd[key].shape)} dtype={sd[key].dtype}")

# ---------- action (7 dim) ----------
for suffix, fn in [
    ("mean", lambda v: pad(v, 0.0)),
    ("std",  pad_std),
    ("q01",  lambda v: pad(v, 0.0)),
    ("q99",  lambda v: pad(v, 0.0)),
]:
    key = f"action_preprocessor.normalizer_action.{suffix}.{NAME}"
    sd[key] = fn(action[suffix])
    print(f"  + {key}  shape={tuple(sd[key].shape)} dtype={sd[key].dtype}")

# ---------- delta ----------
for prefix in ("normalizer_action", "normalizer_propri"):
    base = f"action_preprocessor.{prefix}"
    sd[f"{base}.delta.{NAME}"] = sd[f"{base}.q99.{NAME}"] - sd[f"{base}.q01.{NAME}"]
    print(f"  + {base}.delta.{NAME}  shape={tuple(sd[f'{base}.delta.{NAME}'].shape)}")

save_file(sd, OUT)
print(f"\n✅ saved: {OUT}")
print(f"   total keys: {len(sd)}")

for k in sorted(sd.keys()):
    if f".{NAME}" in k:
        print("   verify:", k, tuple(sd[k].shape), sd[k].dtype)
