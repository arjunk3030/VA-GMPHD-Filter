import numpy as np
from PIL import Image


def display_images_horizontally(images):
    image_arrays = [np.array(im) for im in images]

    stacked_columns = []
    total_height = 0
    max_width = 0

    for im_array in image_arrays:
        width, height = im_array.shape[1], im_array.shape[0]
        max_width = max(max_width, width)
        if total_height + height > 600:
            stacked_image_column = np.hstack(stacked_columns)
            stacked_columns = [stacked_image_column]
            total_height = 0
        stacked_columns.append(im_array)
        total_height += height

    stacked_image_column = np.hstack(stacked_columns)
    stacked_image = Image.fromarray(stacked_image_column)
    # stacked_image.show()  # TODO: ADD BACK


def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = "Input shall be a HxWx3 ndarray"
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, "RGB")
    return img
