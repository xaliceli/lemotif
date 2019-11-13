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
            border_shape=False, border_color=None, inc_floor=0, inc_ceiling=1, text=True, **kwargs):
    if not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    if not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    base_size = int(min(size) * icon_ratio)
    icons_resized = [cv2.resize(icons[topic], (base_size, base_size)) for topic in topics]
    colors_list = [colors[emotion] for emotion in emotions]
    color_icons = [fill_color(icon, color['rgb'], border_color) for icon, color in product(icons_resized, colors_list)]

    canvas = np.zeros((size[0], size[1], 3))
    canvas[..., :] = background
    for p in range(passes):
        complete = False
        start = [random.randint(0, base_size // 4), random.randint(0, base_size // 4)]
        while not complete:
            icon, icon_size = random.choice(color_icons), max(1, int(base_size * np.random.normal(1, size_flux)))
            icon_resized = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_NEAREST)
            adj_y = min(int(start[0] * np.random.normal(1, .025)), size[0] - icon_size)
            mask = bg_mask(icon_resized, colors_list, border_color) if mask_all else None
            increment = random.randint(int(inc_floor * icon_size), int(icon_size * inc_ceiling))
            alpha = random.random() if rand_alpha else 0
            if start[1] + icon_size < canvas.shape[1]:
                canvas = fill_canvas(canvas, background, mask, size, icon_resized, start, adj_y, icon_size, alpha)
                start[1] += increment
            elif start[0] + icon_size < canvas.shape[0]:
                start[0] += increment
                start[1] = increment
                canvas = fill_canvas(canvas, background, mask, size, icon_resized, start, adj_y, icon_size, alpha)
            else:
                complete = True

    if border_shape:
        canvas = apply_shape(canvas, icons, topics, size, border_color, background)

    if text:
        canvas = add_labels(canvas, topics, emotions, colors)

    return canvas