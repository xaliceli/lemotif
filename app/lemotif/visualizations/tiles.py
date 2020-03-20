"""
tile.py
Tile visualization
"""
import gc
import cv2
import numpy as np
from skimage import measure
import random

from lemotif.visualizations.utils import apply_shape, add_labels


def tiles(topics, emotions, icons, colors, size, background=(255, 255, 255),
          border_shape=True, border_color=None, text=True,
          line_width_tile=1, step_size=10, dir_prob=0.5, **kwargs):
    if len(topics) == 0 or len(emotions) == 0:
        return None
    elif topics[0] is None:
        border_shape = False
    elif not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    elif not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    colors_list = [colors[emotion] for emotion in emotions]

    # Draw lines
    canvas = np.ones((size[0], size[1], 3))
    for i in range(0, size[0], step_size):
        for j in range(0, size[1], step_size):
            if random.uniform(0, 1) > dir_prob:
                cv2.line(canvas, (j, i), (j+step_size, i+step_size), (0, 0, 0), line_width_tile)
            else:
                cv2.line(canvas, (j, i+step_size), (j+step_size, i), (0, 0, 0), line_width_tile)
    canvas = np.uint8(canvas[..., 0])

    # Find connected components
    connections = measure.regionprops(measure.label(canvas, connectivity=1))

    # Canvas for output
    canvas = np.zeros((size[0], size[1], 3))
    canvas[..., :] = background

    for component in connections:
        fill = random.choice(colors_list)['rgb']
        canvas[component.coords[:, 0], component.coords[:, 1]] = [fill[2], fill[1], fill[0]]
    del connections
    gc.collect()

    if border_shape:
        canvas = apply_shape(canvas, icons, topics, size, border_color, background)

    if text:
        canvas = add_labels(canvas, topics, emotions, colors)

    return canvas
