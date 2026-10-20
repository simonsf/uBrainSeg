from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from ..utils import ANISO_THRESHOLD, resize_segmentation, timer


def resize_torch(
    slice_: np.ndarray,
    new_shape,
    order=0, **kwargs
) -> np.ndarray:

    mode = "bicubic" if slice_.ndim == 2 else "trilinear"

    slice_ = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0)
    q = F.interpolate(slice_, size=list(new_shape), mode=mode)
    q.squeeze_(0).squeeze_(0)

    return q.numpy()


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


@timer
def resample_data_or_seg_to_shape(data: np.ndarray,
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
                                  faster: bool = False,
                                  device: torch.device = torch.device('cpu')):

    if max(current_spacing) <= 1 and faster:
        separate_z_anisotropy_threshold = 5
        logger.info("Anisotropy threshold change to 5 due to max(current_spacing) <= 1...")

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
        elif len(axis) == 2:
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z, faster=faster, device=device)
    return data_reshaped


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0, faster: bool = False,
                         device: torch.device = torch.device('cpu')):
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize_torch if faster else resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []

            for c in range(data.shape[0]):

                reshaped_data = []
                reshaped_data_ph = np.empty((shape[0], *new_shape[1:]), dtype=np.float32)

                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs))

                np.stack(reshaped_data, axis, out=reshaped_data_ph)

                if shape[axis] != new_shape[axis]:

                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data_ph.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    if (not is_seg or order_z == 0) and faster:

                        matrix = torch.tensor(
                            [
                                [1 / row_scale, 0, 0, 0],
                                [0, 1 / col_scale, 0, 0],
                                [0, 0, 1 / dim_scale, 0]
                            ], dtype=torch.float32
                        ).unsqueeze(0).to(device)

                        grid = F.affine_grid(matrix, torch.Size((1, 1, rows, cols, dim)), align_corners=True)

                        t = torch.from_numpy(reshaped_data_ph).unsqueeze(0).unsqueeze(0).to(device)

                        r = F.grid_sample(t, grid, mode='nearest', align_corners=True)

                        reshaped_final_data.append(r.squeeze(0).cpu().numpy())

                    elif (not is_seg or order_z == 0):

                        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                        map_rows = row_scale * (map_rows + 0.5) - 0.5
                        map_cols = col_scale * (map_cols + 0.5) - 0.5
                        map_dims = dim_scale * (map_dims + 0.5) - 0.5

                        coord_map = np.array([map_rows, map_cols, map_dims])
                        if not is_seg or order_z == 0:
                            reshaped_final_data.append(map_coordinates(reshaped_data_ph, coord_map, order=order_z,
                                                                    mode='nearest')[None])
                    else:

                        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                        map_rows = row_scale * (map_rows + 0.5) - 0.5
                        map_cols = col_scale * (map_cols + 0.5) - 0.5
                        map_dims = dim_scale * (map_dims + 0.5) - 0.5

                        coord_map = np.array([map_rows, map_cols, map_dims])
                        unique_labels = np.sort(np.unique(reshaped_data_ph.ravel()))
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data_ph == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data_ph[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(torch.from_numpy(resize_fn(data[c], new_shape, order, **kwargs)[None]))
            reshaped_final_data = torch.cat(reshaped, dim=0).numpy()
        return reshaped_final_data
    else:
        return data
