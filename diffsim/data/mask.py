"""
Mask generation utilities for inpainting.

Supports various mask types: bbox, irregular, free-form, etc.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def random_cropping_bbox(img_shape=(256, 256), mask_mode='onedirection'):
    """
    Generate random cropping bbox for uncropping task.

    Args:
        img_shape: Image shape (H, W)
        mask_mode: 'onedirection' or 'fourdirection'

    Returns:
        Tuple (top, left, height, width)
    """
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width = 0, 0, h, w // 2
        elif _type == 1:
            top, left, height, width = 0, 0, h // 2, w
        elif _type == 2:
            top, left, height, width = h // 2, 0, h // 2, w
        elif _type == 3:
            top, left, height, width = 0, w // 2, h, w // 2
    else:
        target_area = (h * w) // 2
        width = np.random.randint(target_area // h, w)
        height = target_area // width
        if h == height:
            top = 0
        else:
            top = np.random.randint(0, h - height)
        if w == width:
            left = 0
        else:
            left = np.random.randint(0, w - width)
    return (top, left, height, width)


def random_bbox(img_shape=(256, 256), max_bbox_shape=(128, 128), max_bbox_delta=40, min_margin=20):
    """
    Generate a random bbox for the mask on a given image.

    Args:
        img_shape: The size of the image (h, w)
        max_bbox_shape: Maximum shape of the mask box (h, w)
        max_bbox_delta: Maximum delta of the mask box
        min_margin: Minimum margin size from the edges

    Returns:
        Tuple (top, left, h, w)
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)

    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img shape {img_shape} and mask shape {max_bbox_shape}')

    # Get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w

    # Randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)

    # Randomly shrink the shape of mask box
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """
    Generate mask in ndarray from bbox.

    Args:
        img_shape: The size of the image (H, W)
        bbox: Configuration tuple (top, left, height, width)
        dtype: Data type of returned masks

    Returns:
        Mask in the shape of (h, w, 1)
    """
    height, width = img_shape[:2]
    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1
    return mask


def brush_stroke_mask(img_shape,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(12, 40),
                      max_loops=4,
                      dtype='uint8'):
    """
    Generate free-form mask using brush strokes.

    Based on: Free-Form Image Inpainting with Gated Convolution.

    Args:
        img_shape: Size of the image (H, W)
        num_vertices: Min and max number of vertices
        mean_angle: Mean value of angle at each vertex
        angle_range: Range of random angle
        brush_width: (min_width, max_width)
        max_loops: Max number of stroke loops
        dtype: Data type of returned masks

    Returns:
        Mask in the shape of (h, w, 1)
    """
    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError(f'The type of num_vertices should be int or tuple[int], but got: {type(num_vertices)}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError(f'The type of brush_width should be int or tuple[int], but got: {type(brush_width)}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        vertex = []

        # Set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # Set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        # Draw brush strokes
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2), fill=1)

    # Randomly flip the mask
    if np.random.normal() > 0:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]
    return mask


def random_irregular_mask(img_shape,
                          num_vertices=(4, 8),
                          max_angle=4,
                          length_range=(10, 100),
                          brush_width=(10, 40),
                          dtype='uint8'):
    """
    Generate random irregular masks.

    Args:
        img_shape: Size of the image (H, W)
        num_vertices: Min and max number of vertices
        max_angle: Max value of angle at each vertex
        length_range: (min_length, max_length)
        brush_width: (min_width, max_width)
        dtype: Data type of returned masks

    Returns:
        Mask in the shape of (h, w, 1)
    """
    if not HAS_CV2:
        # Fallback to brush stroke mask if cv2 not available
        return brush_stroke_mask(img_shape, num_vertices=num_vertices,
                                 brush_width=brush_width, dtype=dtype)

    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=dtype)

    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError(f'The type of length_range should be int or tuple[int], but got: {type(length_range)}')

    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError(f'The type of num_vertices should be int or tuple[int], but got: {type(num_vertices)}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError(f'The type of brush_width should be int or tuple[int], but got: {type(brush_width)}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(min_brush_width, max_brush_width, size=direction_num)

        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y

    mask = np.expand_dims(mask, axis=2)
    return mask


def get_irregular_mask(img_shape, area_ratio_range=(0.15, 0.5), **kwargs):
    """
    Get irregular mask with constraints on area ratio.

    Args:
        img_shape: Size of the image (H, W)
        area_ratio_range: (min_ratio, max_ratio) of mask area

    Returns:
        Mask in the shape of (h, w, 1)
    """
    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) / (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask
