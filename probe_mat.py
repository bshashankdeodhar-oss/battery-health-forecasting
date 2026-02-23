"""
probe_mat.py — Print all field names and shapes for XJTU .mat files.
Run: python probe_mat.py
"""
import scipy.io
import numpy as np

PATH = r"Battery Dataset\Battery Dataset\Batch-1\2C_battery-1.mat"

print(f"Loading: {PATH}\n")
mat = scipy.io.loadmat(PATH, squeeze_me=True, struct_as_record=False, mat_dtype=True)
keys = [k for k in mat if not k.startswith("__")]
print(f"Top-level keys: {keys}\n")

for k in keys:
    v = mat[k]
    print(f"=== {k!r} ===")
    if isinstance(v, np.ndarray) and v.dtype == object:
        v = v.item() if v.shape == () else v.flat[0]

    if hasattr(v, "_fieldnames"):
        print(f"  STRUCT with {len(v._fieldnames)} fields:")
        for fn in v._fieldnames:
            child = getattr(v, fn)
            if isinstance(child, np.ndarray):
                print(f"    {fn:40s}  ndarray  shape={child.shape}  dtype={child.dtype}")
                if child.size > 0 and child.dtype != object:
                    print(f"      first 5 vals: {child.flat[:5].tolist()}")
            else:
                print(f"    {fn:40s}  {type(child).__name__}  = {repr(child)[:60]}")
    elif isinstance(v, np.ndarray):
        print(f"  ndarray  shape={v.shape}  dtype={v.dtype}")
        if v.size > 0 and v.dtype == object:
            item = v.flat[0]
            if hasattr(item, "_fieldnames"):
                print(f"  Element[0] is STRUCT, fields={list(item._fieldnames)}")
                for fn in item._fieldnames:
                    child = getattr(item, fn)
                    if isinstance(child, np.ndarray):
                        print(f"    {fn:40s}  shape={child.shape}  dtype={child.dtype}  sample={child.flat[:3].tolist()}")
                    else:
                        print(f"    {fn:40s}  {type(child).__name__}  = {repr(child)[:60]}")
    else:
        print(f"  {type(v).__name__}  = {repr(v)[:80]}")
    print()
