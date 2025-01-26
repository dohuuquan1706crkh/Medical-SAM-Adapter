from typing import Tuple
from numpy.typing import NDArray
import random
import numpy as np
import nibabel as nib
import torch

__all__ = [
    "random_box",
    "random_click",
    "generate_click_prompt",
    "load_nii",
    "apply_dicom_window",
    "get_freq_histogram_bins",
    "scale_image_by_histogram",
    "create_nchannel_map"
]


def random_box(masks: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Get random bboxes
    Args:
        masks (Tensor): masks evaluated by multiple subjects.
    Returns:
        (Tuple): return top-left and bottom-right coordinates (x_min, x_max, y_min, y_max).
    """
    max_value = torch.max(masks[:, 0, :, :], dim=0)[0]
    max_value_position = torch.nonzero(max_value)
    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]
    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))
    x_min = random.choice(np.arange(x_min - 10, x_min + 11))
    x_max = random.choice(np.arange(x_max - 10, x_max + 11))
    y_min = random.choice(np.arange(y_min - 10, y_min + 11))
    y_max = random.choice(np.arange(y_max - 10, y_max + 11))
    return x_min, x_max, y_min, y_max



def random_click(mask: NDArray, point_labels: int = 1) -> Tuple[int, NDArray]:
    """
    Get random click points.

    Args:
        mask (NDArray): a binary or multi-class mask that represents the ground truth or current segmentation state.
        point_labels (int): labels corresponding to each click (point),
            indicating whether the click is on the foreground (1 - positive) or background (0 - negative).
    Returns:
        (int):  label for each clicking point.
        (NDArray): list of positions for the clicks.
    """
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label)
    return point_labels, indices[np.random.randint(len(indices))]


def generate_click_prompt(img: torch.Tensor, msk: torch.Tensor):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)
    msk = msk.unsqueeze(1)
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]


def load_nii(file_path: str) -> NDArray:
    """
    Load CT scan data from *.nii format.

    Args:
        file_path (str): path to *.nii file.
    Returns:
        (NDArray): CT image data.
    """
    ct_scan = nib.load(file_path)
    array = ct_scan.get_fdata()
    array = np.rot90(array)
    return array


def apply_dicom_window(x: NDArray, level: int = 60, width: int = 150) -> NDArray:
    """
    Apply dicom window to image slice to enhance contrast.

    Args:
        x (NDArray): image slice.
        level (int): dicom window level.
        width (int): dicom window width.
    Returns:
        (NDArray): enhanced image slice with normalized value.
    """
    min_value = level - width // 2
    max_value = level + width // 2
    x = np.clip(x, min_value, max_value)
    return (x - min_value) / (max_value - min_value)

def get_freq_histogram_bins(x: NDArray, n_bins: int = 100) -> NDArray:
    """
    Get frequency histogram bins to split the range of pixel values into groups,
    such that each group has around the same number of pixels.

    Args:
        x (NDArray): image slice.
        n_bins (int): number of bins.
    Returns:
        (NDArray): frequency histogram bins.
    """
    imsd = np.sort(x.flatten())
    t = np.concatenate(
     ([0.001],
      np.arange(n_bins) / n_bins + (1 / (2 * n_bins)),
      [0.999])
    )
    t_indices = (len(imsd) * t).astype(int)
    return np.unique(imsd[t_indices])


def scale_image_by_histogram(x: NDArray, brks: NDArray | None = None) -> NDArray:
    """
    Scale image using histogram bins to values between 0 and 1.

    Args:
        x (NDArray): image slice.
        brks (NDArray): histogram bin edges to use for scaling.
    Returns:
        (NDArray): scaled image with normalized value from 0 to 1.
    """
    # compute bin edges
    if brks is None:
        brks = get_freq_histogram_bins(x)
    # generate equally spaced values between 0 and 1 corresponding to the bin edges
    ys = np.linspace(0.0, 1.0, len(brks))
    # flatten and input and perform linear interpolation
    x_flat = x.flatten()
    x_scaled = np.interp(x_flat, brks, ys)
    # reshape and clamp values between 0 and 1
    x_scaled = np.clip(x_scaled.reshape(x.shape), 0.0, 1.0)
    return x_scaled

def create_nchannel_map(x: NDArray, windows: list, bins: int | None = None) -> NDArray:
    """
    Apply windowing and histogram scaling on input data to generate multiple channels.

    Args:
        x (NDArray): image slice.
        windows (list): list of dicom windowing parameters for each channel.
        bins (int): number of bins to use for scaling.
    Returns:
        (NDArray): stacked multi-channel image.
    """
    # apply windowing function for each specified window
    res = [apply_dicom_window(x, *window) for window in windows]
    #  apply histogram scaling
    if not isinstance(bins, int) or bins != 0:
        hist_scaled = scale_image_by_histogram(x, bins)
        res.append(np.clip(hist_scaled, 0, 1))
    return np.stack(res, axis=-1)
