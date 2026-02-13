import numpy as np

def hu_window(ct, hu_min, hu_max):
    ct = ct.astype(np.float32, copy=False)

    hu_min = np.float32(hu_min)
    hu_max = np.float32(hu_max)

    np.clip(ct, hu_min, hu_max, out=ct)
    ct -= hu_min
    ct /= (hu_max - hu_min)

    return ct

def z_score(ct):
    ct = ct.astype(np.float32, copy=False)

    mean = np.float32(ct.mean())
    std = np.float32(ct.std()) + np.float32(1e-8)

    ct -= mean
    ct /= std

    return ct
