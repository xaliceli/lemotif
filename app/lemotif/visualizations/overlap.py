"""
overlap.py
Overlap visualization.
"""
from itertools import product

import cv2
import numpy as np
import random

from lemotif.visualizations.utils import fill_color, bg_mask, fill_canvas, apply_shape, add_labels


def overlap(topics, emotions, icons, colors, size,
            background=(255, 255, 255), icon_ratio=0.1, size_flux=0.25, rand_alpha=True, passes=10, mask_all=True,
            border_shape=False, border_color=None, inc_floor=.5, inc_ceiling=.75, text=True, **kwargs):
    """
    Overlap visualization.

    :param topics: Topics to use (list).
    :param emotions: Emotions to use (list).
    :param icons: Shape icons (dict).
    :param colors: Emotion colors (dict).
    :param background: Background color in RGB (tuple).
    :param border_shape: Whether to apply icon shape as border (bool).
    :param border_color: Border color in RGB (tuple).
    :param text: Include text labels below visualization (bool).
    :param icon_ratio: Ratio of small shape sizes to overall image size (float).
    :param size_flux: Standard deviation by which small shapes vary in size (float).
    :param rand_alpha: Randomly adjust opacity of each shape placed (float).
    :param passes: Number of overlapping passes (int).
    :param mask_all: Opacity mask for individual small shapes (bool).
    :param inc_floor: Minimum step size in placing new shape (float).
    :param inc_ceiling: Maximum step size in placing new shape (float).
    :param kwargs: Additional arguments (dict).
    :return: Visualization (array).
    """
    if len(topics) == 0 or len(emotions) == 0:
        return None
    elif topics[0] is None:
        border_shape = False
    elif not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    elif not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    start_size = (500, 500) # Start here because smaller values run into issues with sub-shape masking
    base_size = int(min(start_size) * icon_ratio)
    if topics[0] is None:
        icons_resized = np.ones((base_size, base_size)) * 255
        border_points = np.concatenate(
            (np.column_stack((np.zeros(base_size), np.array(np.arange(0, base_size)))),
             np.column_stack((np.ones(base_size)*(base_size-1), np.array(np.arange(0, base_size)))),
             np.column_stack((np.array(np.arange(0, base_size)), np.zeros(base_size))),
             np.column_stack((np.array(np.arange(0, base_size)), np.ones(base_size)*(base_size-1))))).astype(int)
        icons_resized[border_points[:, 0], border_points[:, 1]] = 0
        icons_resized = [icons_resized]
    else:
        icons_resized = [cv2.resize(icons[topic], (base_size, base_size)) for topic in topics]
    colors_list = [colors[emotion] for emotion in emotions]
    color_icons = [fill_color(icon, color['rgb'], border_color) for icon, color in product(icons_resized, colors_list)]

    canvas = np.zeros((start_size[0], start_size[1], 3))
    canvas[..., :] = background
    for p in range(passes):
        complete = False
        start = [random.randint(0, base_size // 4), random.randint(0, base_size // 4)]
        while not complete:
            icon, icon_size = random.choice(color_icons), max(1, int(base_size * np.random.normal(1, size_flux)))
            icon_resized = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_NEAREST)
            adj_y = min(int(start[0] * np.random.normal(1, .025)), start_size[0] - icon_size)
            mask = bg_mask(icon_resized, colors_list, border_color) if mask_all else None
            increment = random.randint(int(inc_floor * icon_size), int(icon_size * inc_ceiling))
            alpha = random.random() if rand_alpha else 0
            if start[1] + icon_size < canvas.shape[1]:
                canvas = fill_canvas(canvas, background, mask, start_size, icon_resized, start, adj_y, icon_size, alpha)
                start[1] += increment
            elif start[0] + icon_size < canvas.shape[0]:
                start[0] += increment
                start[1] = increment
                canvas = fill_canvas(canvas, background, mask, start_size, icon_resized, start, adj_y, icon_size, alpha)
            else:
                complete = True

    if border_shape:
        canvas = apply_shape(canvas, icons, topics, start_size, border_color, background)

    canvas = cv2.resize(canvas, size)

    if text:
        canvas = add_labels(canvas, topics, emotions, colors)

    return canvas