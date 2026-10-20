"""Geometric and intensity augmentation operators of the ISM.

Operators are CPU callables on (D, H, W) arrays; the same sampled transform
is applied to all volumes passed jointly.
"""

import numpy as np
from scipy import ndimage


def blurring_sigma_for_downsampling(current_res, target_res):
    """Per-axis Gaussian sigma applied before downsampling; 0 for unchanged axes."""
    current_res = np.array(current_res, dtype=float)
    target_res = np.array(target_res, dtype=float)
    sigma = np.maximum(target_res / (2.0 * current_res), 0.5 * current_res)
    sigma[target_res == current_res] = 0.0
    return sigma


def _rotation_matrix(angles_deg):
    """3-D rotation matrix from angles (degrees): Rz @ Ry @ Rx."""
    angles = np.deg2rad(angles_deg)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _sample_affine_matrix(shape, scaling_bounds, rotation_bounds,
                          shearing_bounds, translation_bounds):
    """Sample a random affine matrix (4, 4) in voxel space, centred on the volume."""
    n_dims = 3
    centre = np.array(shape) / 2.0

    # Scaling
    if scaling_bounds is not False and scaling_bounds is not None:
        sb = float(scaling_bounds) if np.ndim(scaling_bounds) == 0 else scaling_bounds
        if np.ndim(sb) == 0:
            scale = np.random.uniform(1 - sb, 1 + sb, size=n_dims)
        else:
            scale = np.array([np.random.uniform(1 - sb[i], 1 + sb[i]) for i in range(n_dims)])
    else:
        scale = np.ones(n_dims)

    # Rotation
    if rotation_bounds is not False and rotation_bounds is not None:
        rb = float(rotation_bounds) if np.ndim(rotation_bounds) == 0 else rotation_bounds
        if np.ndim(rb) == 0:
            angles = np.random.uniform(-rb, rb, size=3)
        else:
            angles = np.array([np.random.uniform(-rb[i], rb[i]) for i in range(len(rb))])
        R = _rotation_matrix(angles)
    else:
        R = np.eye(n_dims)

    # Shearing
    if shearing_bounds is not False and shearing_bounds is not None:
        sb = float(shearing_bounds) if np.ndim(shearing_bounds) == 0 else shearing_bounds
        Sh = np.eye(n_dims)
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j:
                    Sh[i, j] = np.random.uniform(-sb, sb)
    else:
        Sh = np.eye(n_dims)

    A = np.diag(scale) @ R @ Sh

    # Translation
    if translation_bounds is not False and translation_bounds is not None:
        tb = float(translation_bounds) if np.ndim(translation_bounds) == 0 else translation_bounds
        t = np.random.uniform(-tb, tb, size=n_dims)
    else:
        t = np.zeros(n_dims)

    M = np.eye(n_dims + 1)
    M[:n_dims, :n_dims] = A
    M[:n_dims, n_dims] = (np.eye(n_dims) - A) @ centre + t
    return M


class SpatialDeformation:
    """Random affine + elastic (nonlinear) spatial deformation."""

    def __init__(self,
                 scaling_bounds=0.1,
                 rotation_bounds=15.0,
                 shearing_bounds=0.02,
                 translation_bounds=False,
                 nonlin_std=2.0,
                 nonlin_scale=0.04,
                 inter_method='nearest'):
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.inter_method = inter_method

    def __call__(self, *volumes):
        """Apply the same random deformation to all volumes, return list."""
        shape = volumes[0].shape[:3]
        n_dims = 3
        flow = np.zeros(list(shape) + [n_dims], dtype=np.float32)

        # Affine component
        apply_affine = any([
            self.scaling_bounds is not False and self.scaling_bounds is not None,
            self.rotation_bounds is not False and self.rotation_bounds is not None,
            self.shearing_bounds is not False and self.shearing_bounds is not None,
            self.translation_bounds is not False and self.translation_bounds is not None,
        ])
        if apply_affine:
            M = _sample_affine_matrix(shape, self.scaling_bounds, self.rotation_bounds,
                                      self.shearing_bounds, self.translation_bounds)
            coords = np.mgrid[tuple(slice(0, s) for s in shape)]
            coords_flat = coords.reshape(n_dims, -1)
            ones = np.ones((1, coords_flat.shape[1]))
            coords_h = np.vstack([coords_flat, ones])
            new_coords_flat = (M @ coords_h)[:n_dims]
            aff_flow = (new_coords_flat - coords_flat).T.reshape(list(shape) + [n_dims])
            flow += aff_flow.astype(np.float32)

        # Elastic component: stationary velocity field integrated by scaling-and-squaring
        if self.nonlin_std > 0:
            small = [max(int(s * self.nonlin_scale), 1) for s in shape]
            std = np.random.uniform(0, self.nonlin_std)
            svf_small = np.random.normal(0, std, size=small + [n_dims]).astype(np.float32)
            half = [max(s // 2, sv) for s, sv in zip(shape, small)]
            svf_half = self._resize(svf_small, half)
            svf_int = self._integrate_svf(svf_half)
            svf_full = self._resize(svf_int, list(shape))
            flow += svf_full

        modes = getattr(self, '_per_vol_modes', None)
        out = []
        for i, vol in enumerate(volumes):
            m = modes[i] if modes is not None and i < len(modes) else self.inter_method
            order = 1 if m == 'linear' else 0
            out.append(self._warp(vol, flow, order))
        return out if len(out) > 1 else out[0]

    def warp_volumes(self, *volumes, inter_methods=None):
        """Like :meth:`__call__` but with per-volume interpolation modes."""
        self._per_vol_modes = inter_methods
        try:
            return self(*volumes)
        finally:
            self._per_vol_modes = None

    @staticmethod
    def _resize(field, target_shape):
        factors = [t / s for t, s in zip(target_shape, field.shape[:3])] + [1]
        return ndimage.zoom(field, factors, order=1)

    @staticmethod
    def _integrate_svf(svf, n_steps=7):
        flow = svf / (2 ** n_steps)
        for _ in range(n_steps):
            flow = flow + SpatialDeformation._warp(flow, flow, order=1)
        return flow

    @staticmethod
    def _warp(vol, flow, order=1):
        """Warp vol (D,H,W) or (D,H,W,C) by displacement field flow (D,H,W,3)."""
        shape = vol.shape[:3]
        coords = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float32)
        new_coords = coords + flow.transpose(3, 0, 1, 2)

        if vol.ndim == 3:
            return ndimage.map_coordinates(vol, new_coords, order=order,
                                           mode='nearest').astype(vol.dtype)
        # 4-D: warp each displacement channel separately (used by SVF integration)
        channels = [ndimage.map_coordinates(vol[..., c], new_coords, order=order,
                                            mode='nearest').astype(vol.dtype)
                    for c in range(vol.shape[3])]
        return np.stack(channels, axis=-1)


class RandomCrop:
    """Randomly crop spatial dimensions to ``crop_shape`` (same window for all volumes)."""

    def __init__(self, crop_shape):
        self.crop_shape = list(crop_shape)

    def __call__(self, *volumes):
        shape = volumes[0].shape[:3]
        starts = [np.random.randint(0, max(shape[i] - self.crop_shape[i], 1))
                  for i in range(3)]
        slices = tuple(slice(s, s + c) for s, c in zip(starts, self.crop_shape))
        out = [vol[slices] for vol in volumes]
        return out if len(out) > 1 else out[0]


class RandomFlip:
    """Randomly flip volumes along the first array axis with probability ``prob``."""

    def __init__(self, prob=0.5, axis=0):
        self.prob = prob
        self.axis = axis

    def __call__(self, *volumes):
        if np.random.random() >= self.prob:
            out = list(volumes)
        else:
            out = [np.flip(vol, axis=self.axis).copy() for vol in volumes]
        return out if len(out) > 1 else out[0]


class BiasField:
    """Multiplicative smooth random bias field: image * exp(zoomed Gaussian noise)."""

    def __init__(self, std=0.2, scale=0.025):
        self.std = std
        self.scale = scale  # ratio: small field size / image size

    def __call__(self, image):
        shape = image.shape[:3]
        small = [max(int(s * self.scale), 1) for s in shape]
        field_small = np.random.normal(0, self.std, size=small).astype(np.float32)
        factors = [s / sm for s, sm in zip(shape, small)]
        field = ndimage.zoom(field_small, factors, order=1)
        field = np.exp(field)
        if image.ndim == 4:
            field = field[..., np.newaxis]
        return (image * field).astype(image.dtype)


class IntensityAugmentation:
    """Clip -> normalise -> gamma -> additive noise."""

    def __init__(self, clip=False, normalize=True, gamma_std=0.5, noise_std=0.0):
        self.clip = clip
        self.normalize = normalize
        self.gamma_std = gamma_std
        self.noise_std = noise_std

    def __call__(self, image):
        if self.clip is not False:
            image = np.clip(image, 0, self.clip).astype(np.float32)
        if self.normalize:
            mn, mx = image.min(), image.max()
            if mx > mn:
                image = (image - mn) / (mx - mn)
        if self.gamma_std > 0:
            gamma = np.exp(np.random.normal(0, self.gamma_std))
            image = np.power(np.clip(image, 0, None), gamma).astype(np.float32)
        if self.noise_std > 0:
            image = (image + np.random.normal(0, self.noise_std, image.shape)).astype(np.float32)
        return image


class GaussianBlur:
    """Separable Gaussian blur with per-axis sigma."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        sigma = self.sigma
        if not hasattr(sigma, '__len__'):
            sigma = [sigma] * 3
        for ax, s in enumerate(sigma):
            if s > 0:
                image = ndimage.gaussian_filter1d(image, sigma=s, axis=ax)
        return image


class SimulateAcquisition:
    """Resolution variability: Gaussian blur -> downsample -> upsample back."""

    def __init__(self, current_res, target_res):
        self.current_res = np.array(current_res, dtype=float)
        self.target_res = np.array(target_res, dtype=float)

    def __call__(self, image):
        sigma = blurring_sigma_for_downsampling(self.current_res, self.target_res)

        blurred = image.copy()
        for ax, s in enumerate(sigma):
            if s > 0:
                blurred = ndimage.gaussian_filter1d(blurred, sigma=s, axis=ax)

        zoom_down = self.current_res / self.target_res
        down = ndimage.zoom(blurred, list(zoom_down) + [1] * (blurred.ndim - 3), order=1)

        zoom_up = np.array(image.shape[:3]) / np.array(down.shape[:3])
        up = ndimage.zoom(down, list(zoom_up) + [1] * (down.ndim - 3), order=1)

        out = np.zeros_like(image)
        slices = tuple(slice(0, min(image.shape[i], up.shape[i])) for i in range(3))
        out[slices] = up[slices]
        return out
