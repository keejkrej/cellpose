import numpy as np
from cellpose.utils import (
    fill_holes_and_remove_small_masks,
    get_mask_ellipse_diameters,
    get_mask_pixel_cv,
)
import fastremap


def test_fill_holes_and_remove_small_masks():
    # make a 2-channel mask with holes and small objects. The first channel is the
    # "ground truth" without holes or small objects, the second channel needs to be cleaned.
    masks = np.zeros((2, 100, 100), dtype=np.uint16)
    masks[:, 10:30, 10:30] = 1  # object 1
    masks[1, 15:25, 15:25] = 0  # hole in object 1
    masks[1, 40:45, 40:45] = 2  # small object 2
    masks[:, 60:90, 60:90] = 4  # object 4 (skip 3)
    masks[1, 70:80, 70:80] = 0  # hole in object 4
    masks[1, 10:15, 80:82] = 5  # small object 4

    # apply function
    min_size = 30
    masks_cleaned = fill_holes_and_remove_small_masks(masks[1], min_size=min_size)

    gt_masks = fastremap.renumber(masks[0], in_place=False)[0]

    assert (gt_masks == masks_cleaned).all()


def test_get_mask_ellipse_diameters():
    masks = np.zeros((40, 60), dtype=np.uint16)
    masks[10:30, 15:45] = 1
    major, minor = get_mask_ellipse_diameters(masks)
    assert major.shape == (1,)
    assert minor.shape == (1,)
    assert major[0] > minor[0]
    assert major[0] > 15
    assert minor[0] > 5

    empty_major, empty_minor = get_mask_ellipse_diameters(np.zeros((10, 10), dtype=np.uint16))
    assert empty_major.shape == (0,)
    assert empty_minor.shape == (0,)


def test_get_mask_pixel_cv():
    image = np.zeros((40, 60), dtype=np.float64)
    image[10:30, 15:45] = 10.0
    image[20:25, 25:35] = 30.0
    masks = np.zeros((40, 60), dtype=np.uint16)
    masks[10:30, 15:45] = 1

    cv = get_mask_pixel_cv(masks, image)
    assert cv.shape == (1,)
    assert cv[0] > 0

    uniform = np.full((40, 60), 5.0, dtype=np.float64)
    uniform_cv = get_mask_pixel_cv(masks, uniform)
    assert uniform_cv[0] == 0.0
