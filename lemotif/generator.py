"""
generator.py
Generates visualization based on algorithm.
"""

import glob
import os

import cv2

from assets.colors import colors_dict
from visualizations.overlap import overlap


def load_assets(input_dir='../assets/'):
    """
    Loads icons and colors.

    Args:
        input_dir (str): Folder where assets reside.
    """
    icons_dict = {}
    for icon in glob.glob(os.path.join(input_dir, 'icons', '*.png')):
        name = os.path.basename(icon)[:-4]
        icons_dict[name] = cv2.imread(icon, -1)

    return icons_dict, colors_dict


def generate_visual(icons, colors, topics, emotions, algorithm, out='../output', size=(500, 500), **args):
    """
    Generates visualization based on inputs.

    Args:
        icons (dict): Dictionary of icons.
        colors (dict): Dictionary of colors.
        topics (list(list)): Nested list of topics. Top level is per visualization.
        emotions (list(list)): Nested list of emotions. Top level is per visualization.
        algorithm (function): Algorithm to generate visual using.
    """
    outputs = [algorithm(t, e, icons, colors, size, **args) for t, e in zip(topics, emotions)]
    if not os.path.isdir(out):
        os.mkdir(out)
    for i, vis in enumerate(outputs):
        cv2.imwrite(os.path.join(out, str(i) + '.png'), vis)
    return outputs


if __name__ == '__main__':
    icons, colors = load_assets()
    topics = [['family', 'food'], ['exercise', 'health'], ['work', 'school']]
    emotions = [['happy', 'satisfied'], ['proud', 'excited'], ['anxious', 'afraid']]

    # Lemotif adjustable settings
    args = {}
    # Canvas size for output
    args['size'] = (500, 500)
    # Canvas background in BGR format
    args['background'] = (255, 255, 255)
    # Base size of icon relative to canvas size
    args['icon_ratio'] = 0.1
    # Standard deviation of a normal distribution centered at 1 from which icon resizing factors are sampled
    args['size_flux'] = 0.33
    # Number of times the canvas is iterated over; smaller numbers retain appearance of shapes, larger numbers appear more painterly
    args['passes'] = 10
    # Minimum incremental distance as factor of icon size before new shape can be placed. Lower results in more overlap.
    args['inc_floor'] = 0.5
    # Maximum incremental distance as factor of icon size before new shape can be placed. Lower results in more overlap.
    args['inc_ceiling'] = 0.75
    # If True, each icon is alpha-blended into the existing canvas at a random opacity
    args['rand_alpha'] = True
    # If True, entire shapes are alpha-blended, otherwise blends only overlap regions
    args['mask_all'] = True
    # If True, randomly select topic for border shape as opposed to using square.
    args['border_shape'] = True
    # Color of icon borders: None for no borders, otherwise scalar representing the relative brightness value of borders
    args['border_color'] = 0.5

    generate_visual(icons, colors, topics, emotions, overlap, **args)