import SimpleITK as sitk
import numpy as np

def load_nifti(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize()
    }

    return arr, img, meta
