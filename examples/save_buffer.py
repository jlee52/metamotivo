import torch
import h5py
import numpy as np
from metamotivo.buffers.buffers import DictBuffer


def save_h5(group, name, value):
    """
    Write `value` under `group[name]`. If `value` is
    a dict, make a subgroup and recurse; if it's a list,
    stack into an array; if it's a tensor, convert to numpy.
    """
    if isinstance(value, dict):
        subgrp = group.create_group(name)
        for k, v in value.items():
            save_h5(subgrp, k, v)
    else:
        # convert Torch tensor → NumPy
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        # if it’s a list of things, try stacking
        elif isinstance(value, list):
            arrs = []
            for v in value:
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                arrs.append(np.array(v))
            arr = np.stack(arrs, axis=0)
        else:
            arr = np.array(value)
        # now arr is a plain numeric array
        group.create_dataset(name, data=arr, dtype=arr.dtype)


path = "/media/mitmotorcontrol/Seagate Portable Drive/jlee/tmp_fbcpr/fbcpr_residual_1024_12_ta/checkpoint/"
fpath = path + "buffer.hdf5"

hf = h5py.File(fpath, "r")
print(hf.keys())
data = {}
for k, v in hf.items():
    if isinstance(v, h5py.Group):
        print(f"{k:20s}: <Group> with keys {list(v.keys())}")
        data[k] = {kk: v[kk][:] for kk in v.keys()}
    else:
        print(f"{k:20s}: {v.shape}")
        data[k] = v[:]
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)
del data


bf = buffer.get_full_buffer()
full_buffer = DictBuffer(capacity=bf["action"].shape[0], device="cpu")

data = {}
data["action"] = bf["action"]
data["next_observation"] = bf["next"]["observation"]
data["next_qpos"] = bf["next"]["qpos"]
data["next_qvel"] = bf["next"]["qvel"]
data["observation"] = bf["observation"]
data["qpos"] = bf["qpos"]
data["qvel"] = bf["qvel"]

full_buffer.extend(data)
del data


# Save
save_path = path + "full_buffer.hdf5"
with h5py.File(save_path, 'w') as hf:
    for key, value in full_buffer.storage.items():
        save_h5(hf, key, value)


sample = full_buffer.sample(500_000)
print(sample.keys())
data = {}
for k, v in sample.items():
    print(f"{k:20s}: {v.shape}")
    data[k] = v[:]
sample_buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
sample_buffer.extend(data)
del data


# Save
save_path = path + "buffer_inference_500000.hdf5"
with h5py.File(save_path, 'w') as hf:
    for key, value in sample_buffer.storage.items():
        save_h5(hf, key, value)