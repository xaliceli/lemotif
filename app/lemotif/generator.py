"""
generator.py
Generates visualization based on algorithm.
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd

from lemotif.colors import colors_dict
from lemotif.visualizations.utils import rgb_to_hsv
from lemotif.visualizations.carpet import carpet
from lemotif.visualizations.circle import circle
from lemotif.visualizations.overlap import overlap as glass
from lemotif.visualizations.string import string
from lemotif.visualizations.tiles import tiles as tile
from lemotif.visualizations.autoencoder import ae as watercolors


def load_assets(input_dir='../assets'):
    """
    Loads icons and colors.

    Args:
        input_dir (str): Folder where assets reside.
    """
    icons_dict = {}
    for icon in glob.glob(os.path.join(input_dir, '*.png')):
        name = os.path.basename(icon)[:-4]
        icons_dict[name] = cv2.imread(icon, -1)
    for color in colors_dict.keys():
        colors_dict[color]['hex'] = '#%02x%02x%02x' % colors_dict[color]['rgb']
        colors_dict[color]['hsv'] = rgb_to_hsv(colors_dict[color]['rgb'])

    return icons_dict, colors_dict


def generate_visual(icons, colors, topics, emotions, algorithm, out_dir=None, size=(500, 500), summary=False,
                    concat=True, all_styles=None, **args):
    """
    Generates visualization based on inputs.

    Args:
        icons (dict): Dictionary of icons.
        colors (dict): Dictionary of colors.
        topics (list(list)): Nested list of topics. Top level is per visualization.
        emotions (list(list)): Nested list of emotions. Top level is per visualization.
        algorithm (function): Algorithm to generate visual using.
    """
    algorithm = globals()[algorithm]
    topics_use = [t for t in topics if t[0] is not None]
    emotions_use = [e for i, e in enumerate(emotions) if topics[i][0] is not None]
    all_out, all_combined = [], np.zeros((size[0]+20, size[1]*len(topics_use), 3))

    for id, (sub_t, sub_e) in enumerate(zip(topics_use, emotions_use)):
        if len(sub_t) > 0:
            sub_t = [sub_t[0]]
        if len(sub_e) > 4:
            sub_e = sub_e[:4]
        if all_styles is not None:
            outputs = [globals()[a](sub_t, sub_e, icons, colors, size, **args) for a in all_styles]
        else:
            outputs = [algorithm(sub_t, sub_e, icons, colors, size, **args)]

        if summary:
            summary_args = args.copy()
            summary_args['border_shape'] = False
            outputs.append(algorithm(
                [item for sublist in topics_use for item in sublist],
                [item for sublist in emotions_use for item in sublist],
                icons, colors, size, **summary_args)
            )

        if out_dir is not None:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if concat:
                final = np.zeros((outputs[0].shape[0], outputs[0].shape[1]*len(sub_t), 3))
                for i, vis in enumerate(outputs):
                    final[:, i*outputs[0].shape[1]:(i+1)*outputs[0].shape[1], :] = vis
                cv2.imwrite(os.path.join(out_dir, str(id) + '.png'), final)
            else:
                for i, vis in enumerate(outputs):
                    cv2.imwrite(os.path.join(out_dir, str(i) + '.png'), vis)

        all_out += outputs
        all_combined[:, id * size[1]:(id + 1) * size[1], :] = outputs[0]

    return all_out, all_combined
