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

    # for color, details in colors_dict.items():
    #     colors_dict[color]['hsv'] = cv2.cvtColor(details['rgb'], cv2.COLOR_BGR2HSV)

    return icons_dict, colors_dict


def generate_visual(icons, colors, topics, emotions, algorithm, out='../output', size=(500, 500)):
    """
    Generates visualization based on inputs.

    Args:
        icons (dict): Dictionary of icons.
        colors (dict): Dictionary of colors.
        topics (list(list)): Nested list of topics. Top level is per visualization.
        emotions (list(list)): Nested list of emotions. Top level is per visualization.
        algorithm (function): Algorithm to generate visual using.
    """
    outputs = [algorithm(t, e, icons, colors, size) for t, e in zip(topics, emotions)]
    if not os.path.isdir(out):
        os.mkdir(out)
    for i, vis in enumerate(outputs):
        cv2.imwrite(os.path.join(out, str(i) + '.png'), vis)
    return outputs


if __name__ == '__main__':
    icons, colors = load_assets()
    topics = [['family', 'food'], ['exercise', 'health'], ['work', 'school']]
    emotions = [['happy', 'satisfied'], ['proud', 'excited'], ['anxious', 'afraid']]
    generate_visual(icons, colors, topics, emotions, overlap)