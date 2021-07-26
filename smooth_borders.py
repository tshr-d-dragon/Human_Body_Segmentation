
### This code is to improve mask borders and taken from this StackOverflow answer:
### https://stackoverflow.com/questions/64928788/make-mask-border-more-distinguished-in-instance-segmentation-mask

from scipy.ndimage.filters import gaussian_filter
import numpy as np


def fix_patch(patch, val):
    patch_tmp = np.where(patch == val, patch, 0)
    blurred_patch = gaussian_filter(patch_tmp, sigma=0.7)
    patch_tmp = np.where((blurred_patch < int(0.9 * val)) & (blurred_patch > int(0.5 * val)), 0, 1)
    return patch * patch_tmp


def smart_matrix_indexing(r_min, r_max, c_min, c_max, mat):
    row_max, col_max = np.subtract(mat.shape, (1, 1))
    r_min = np.max([r_min - 3, 0])
    r_max = np.min([r_max + 3, row_max])
    c_min = np.max([c_min - 3, 0])
    c_max = np.min([c_max + 3, col_max])
    return r_min, r_max, c_min, c_max


def fix_segmentation_maps(mask):
    unique_values = np.unique(mask)
    unique_values = unique_values[np.where(unique_values > 0)]
    for val in unique_values:
        r, c = np.where(mask == val)
        r_min, r_max, c_min, c_max = smart_matrix_indexing(r.min(), r.max(), c.min(), c.max(), mask)
        patch = mask[r_min:r_max, c_min:c_max]
        mask[r_min:r_max, c_min:c_max] = fix_patch(patch, val)
    return mask


# fixed_mask = fix_segmentation_maps(mask)
