"""
circle.py
Circle Packing visualization
"""
import cv2
import numpy as np

from lemotif.visualizations.utils import apply_shape, add_labels, shape_bool_mask


def circle(topics, emotions, icons, colors, size, background=(255, 255, 255), min_rad_factor=.01, max_rad_factor=.09,
           n_circles=100, max_attempts=100, border_shape=True, border_color=None, border_width=1, text=True, **kwargs):
    if len(topics) == 0 or len(emotions) == 0:
        return None
    elif topics[0] is None:
        border_shape = False
    elif not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    elif not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'
    colors_list = [colors[emotion] for emotion in emotions]

    canvas = np.zeros((size[0], size[1], 3))
    canvas[..., :] = background

    if topics[0] is None:
        bool_mask = np.ones((size[0], size[1])).astype(bool)
        min_row, max_row = 0, size[0]-1
        min_col, max_col = 0, size[1]-1
    else:
        # Load a boolean mask indicating whether each coordinate is within desired outline shape
        bool_mask = shape_bool_mask(icons, topics, size, border_color)[..., 0]
        interior_coords = np.argwhere(bool_mask)
        min_row, max_row = np.min(interior_coords[:, 0]), np.max(interior_coords[:, 0])
        min_col, max_col = np.min(interior_coords[:, 1]), np.max(interior_coords[:, 1])
        canvas[bool_mask] = (245, 245, 245)

    # Choose a set of random radii and sort by largest first
    min_rad, max_rad = min_rad_factor*size[0], max_rad_factor*size[0]
    circles_to_place = (min_rad + (max_rad - min_rad) * np.random.random(n_circles) * np.random.random(n_circles)).astype('int')
    circles_to_place[::-1].sort()
    print(circles_to_place)

    # For each circle, X number of attempts to place it without going outside boundary or overlapping another circle
    can_place = bool_mask
    fail_count = 0
    all_rows, all_cols = np.arange(size[0]), np.arange(size[1])
    for r in circles_to_place:
        if fail_count < n_circles/10:
            placed, try_idx = False, 0
            row_try = np.random.uniform(min_row, max_row, max_attempts)
            col_try = np.random.uniform(min_col, max_col, max_attempts)
            try_coords = np.concatenate((row_try[:, None], col_try[:, None]), 1).astype('int')
            while not placed and try_idx < max_attempts:
                try_center = try_coords[try_idx]
                # Get coordinates of circle with given center and radius
                spacing_mask = (all_cols[None, :] - try_center[1]) ** 2 + \
                              (all_rows[:, None] - try_center[0]) ** 2 < (r+border_width) ** 2
                circle_mask = (all_cols[None, :] - try_center[1]) ** 2 + \
                              (all_rows[:, None] - try_center[0]) ** 2 < r ** 2
                inner_mask = (all_cols[None, :] - try_center[1]) ** 2 + \
                             (all_rows[:, None] - try_center[0]) ** 2 < (r-border_width) ** 2
                no_conflicts = np.all(np.logical_and(can_place, spacing_mask)[spacing_mask])
                if no_conflicts:
                    color = np.random.choice(colors_list)['rgb']
                    canvas[circle_mask] = [50, 50, 50]
                    canvas[inner_mask] = [color[2], color[1], color[0]]
                    can_place = np.logical_and(can_place, ~spacing_mask)
                    placed = True
                    fail_count = 0
                else:
                    try_idx += 1
            fail_count += 1

    if border_shape:
        canvas = apply_shape(canvas, icons, topics, size, border_color, background)

    if text:
        canvas = add_labels(canvas, topics, emotions, colors)

    return canvas
