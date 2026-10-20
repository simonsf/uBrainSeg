"""NIfTI I/O for the ISM (SimpleITK-based, geometry-preserving)."""

import os

import numpy as np
import SimpleITK as sitk


def load_mask(path):
    """Load a label map, return (array (D, H, W) int32, sitk image)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    return arr, img


def load_image(path):
    """Load an intensity volume, return (array (D, H, W) float32, sitk image)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def save_volume(array, ref, path):
    """Save ``array`` as NIfTI, copying geometry from ``ref`` (spacing/direction only if shapes differ)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.asarray(array)
    img = sitk.GetImageFromArray(arr.astype(arr.dtype, copy=False))
    if img.GetSize() == ref.GetSize():
        img.CopyInformation(ref)
    else:
        img.SetSpacing(ref.GetSpacing())
        img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, path)
    return path


def mask_voxel_spacing(img):
    """Voxel size (mm) of ``img`` in array axis order (D, H, W)."""
    return tuple(float(s) for s in reversed(img.GetSpacing()))
