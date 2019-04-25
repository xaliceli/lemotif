"""
overlap.py
Overlap visualization.
"""
import cv2
import numpy as np
import random

from visualizations.utils import fill_color

def overlap(topics, emotions, icons, colors, size, rand_color=True, icon_ratio=0.1):
    if not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    if not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    icon_size = int(min(size) * icon_ratio)
    icons = [cv2.resize(icons[topic], (icon_size, icon_size)) for topic in topics]
    colors = [colors[emotion] for emotion in emotions]
    color_icons = [fill_color(icon, color['rgb']) for icon, color in zip(icons, colors)]

    canvas, complete = np.ones((size[0], size[1], 3))*255, False
    start = [random.randint(0, icon_size//2), random.randint(0, icon_size//2)]
    while not complete:
        icon = random.choice(color_icons)
        mask = icon[:, :] != [255, 255, 255]
        canvas[start[0]:start[0] + icon_size, start[1]:start[1] + icon_size][mask] = icon[mask]
        increment = random.randint(0, icon_size)
        if start[1] + increment + icon_size < canvas.shape[1]:
            start[1] += increment
        elif start[0] + increment + icon_size < canvas.shape[0]:
            start[0] += increment
            start[1] = increment
        else:
            complete = True

    return canvas