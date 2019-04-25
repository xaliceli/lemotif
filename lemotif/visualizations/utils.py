"""
utils.py
Visualization helpers.
"""

import cv2
import numpy as np
from scipy.spatial import distance


def fill_color(shape, color, color_type='rgb', bg_opacity=0):
    # TODO: Opacity
    # TODO: Some shapes won't work with this, need to order coords in clockwise order
    """
    Fills in shape delineated by black lines with specified color.
    """
    # Get coordinates of borders
    borders = np.array(np.where(shape == 0)).T
    # Fill in interior of borders with color
    filled = np.ones((shape.shape[0], shape.shape[1], 3))*255
    filled = cv2.fillConvexPoly(filled, borders[:, [1, 0]], color)  # Borders need to be reordered into x, y

    return filled
