"""
carpet.py
Carpet visualization
"""
import gc
import cv2
import numpy as np
from skimage import measure
import random

from lemotif.visualizations.utils import apply_shape, add_labels


def carpet(topics, emotions, icons, colors, size, background=(255, 255, 255),
           border_shape=True, border_color=None, text=True,
           tile_ratio=.1, line_width=1, rotations=4, rot_degree=45, num_lines=3,
           **kwargs):
    if not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    if not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    colors_list = [colors[emotion] for emotion in emotions]

    canvas = np.zeros((size[0], size[1], 3))
    canvas[..., :] = background

    tile_size = int(size[0]*tile_ratio)
    num_tiles = (int(size[0]/tile_size), int(size[1]/tile_size))
    line_array = np.linspace(0, tile_size, num_lines)[:, None]

    for i in range(num_tiles[0]):
        for j in range(num_tiles[1]):
            x_start = j*tile_size
            y_start = i*tile_size
            all_start =  np.int32(line_array + [x_start, y_start])
            angle = random.choice(range(0, rotations*rot_degree, rot_degree))
            for line in range(num_lines):
                if angle == 0:
                    start = (all_start[0, 0], all_start[line, 1])
                    end = (all_start[num_lines-1, 0], all_start[line, 1])
                elif angle == rot_degree:
                    if line < num_lines - 1:
                        start = (all_start[0, 0], all_start[num_lines-1-line, 1])
                        end = (all_start[line+1, 0], all_start[num_lines-1, 1])
                    elif 0 < line < num_lines - 1:
                        start2 = (all_start[line, 0], all_start[0, 1])
                        end2 = (all_start[num_lines-1, 0], all_start[num_lines-line, 1])
                        cv2.line(canvas, start2, end2, (0, 0, 0), line_width)
                elif angle == rot_degree*2:
                    start = (all_start[line, 0], all_start[0, 1])
                    end = (all_start[line, 0], all_start[num_lines-1, 1])
                elif angle == rot_degree*3:
                    if line < num_lines - 1:
                        start = (all_start[0, 0], all_start[line+1, 1])
                        end = (all_start[line+1, 0], all_start[0, 1])
                    elif 0 < line < num_lines - 1:
                        start2 = (all_start[line, 0], all_start[num_lines-1, 1])
                        end2 = (all_start[num_lines-1, 0], all_start[line, 1])
                        cv2.line(canvas, start2, end2, (0, 0, 0), line_width)
                cv2.line(canvas, start, end, (0, 0, 0), line_width)
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
