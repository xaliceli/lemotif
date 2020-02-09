"""
string.py
String Doll visualization
"""
import gc
import cv2
import numpy as np
import random
from scipy.special import comb


from lemotif.visualizations.utils import apply_shape, add_labels, shape_bool_mask


def string(topics, emotions, icons, colors, size, background=(255, 255, 255), n_lines=150, line_width=4,
           border_shape=True, border_color=None, text=True, **kwargs):
    if not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    if not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    colors_list = [colors[emotion] for emotion in emotions]

    canvas = np.zeros((size[0], size[1], 3))
    canvas[..., :] = background

    # Load a boolean mask indicating outline region
    bool_mask = shape_bool_mask(icons, topics, size, border_color)[..., 0]
    canvas[bool_mask] = (245, 245, 245)

    # Get coordinates of shape boundaries to start and end curves
    border = cv2.resize(icons[topics[0]], size) / 255
    border_points = np.argwhere(~border.astype(bool)).tolist()

    # For n number of curves, select arbitrary start and end point on boundary, then two random midpoints
    for line in range(n_lines):
        start, end = random.sample(border_points, 1)[0], random.sample(border_points, 1)[0]
        offset = np.random.normal(scale=.2)
        mid1 = np.array([start[0]*1/3 + end[0]*2/3, start[1]*1/3 + end[1]*2/3])*(1+offset)
        mid2 = np.array([start[0]*2/3 + end[0]*1/3, start[1]*2/3 + end[1]*1/3])*(1+offset)
        points = [start, mid1, mid2, end]

        nPoints = len(points)
        xPoints = np.array([p[1] for p in points])
        yPoints = np.array([p[0] for p in points])

        t = np.linspace(0.0, 1.0, 50)

        polynomial_array = np.array([comb(nPoints-1, i) * ( t**(nPoints-1-i) ) * (1 - t)**i for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)[:, None]
        yvals = np.dot(yPoints, polynomial_array)[:, None]

        curve_points = np.concatenate((xvals, yvals), axis=1).astype(np.int32)[None, ...]

        color = np.random.choice(colors_list)['rgb']

        cv2.polylines(canvas, curve_points, False, [color[2], color[1], color[0]], line_width, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, curve_points, False, [int(color[2]*1.5), int(color[1]*1.5), int(color[0]*1.5)], 1, lineType=cv2.LINE_AA)

    if border_shape:
        canvas = apply_shape(canvas, icons, topics, size, border_color, background)

    if text:
        canvas = add_labels(canvas, topics, emotions, colors)

    return canvas
