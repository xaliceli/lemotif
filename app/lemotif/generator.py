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
from lemotif.visualizations.overlap import overlap
from lemotif.visualizations.string import string
from lemotif.visualizations.tiles import tiles
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
    all_out = []
    for id, (sub_t, sub_e) in enumerate(zip(topics, emotions)):
        sub_t = [sub_t[0]]
        if all_styles is not None:
            outputs = [globals()[a](sub_t, sub_e, icons, colors, size, **args) for a in all_styles]
        else:
            outputs = [algorithm(sub_t, sub_e, icons, colors, size, **args)]

        if summary:
            summary_args = args.copy()
            summary_args['border_shape'] = False
            outputs.append(algorithm(
                [item for sublist in topics for item in sublist],
                [item for sublist in emotions for item in sublist],
                icons, colors, size, **summary_args)
            )
        if out_dir is not None:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if concat:
                final = np.zeros((outputs[0].shape[0], outputs[0].shape[1]*len(sub_t), 3))
                for i, vis in enumerate(outputs):
                    final[:, i*outputs[0].shape[1]:(i+1)*outputs[0].shape[1], :] = vis
                print(os.path.join(out_dir, str(id) + '.png'))
                cv2.imwrite(os.path.join(out_dir, str(id) + '.png'), final)
            else:
                for i, vis in enumerate(outputs):
                    cv2.imwrite(os.path.join(out_dir, str(i) + '.png'), vis)

        all_out += outputs

    return all_out


if __name__ == '__main__':
    import math

    icons, colors = load_assets()

    topics, emotions = [], []
    data_path = '/Users/alice/Google Drive/Colab Data/lemotif/data/input_100.csv'
    df = pd.read_csv(data_path)

    for i, row in df.iterrows():
        f_use, t_use = [], []
        for set in range(3):
            if not isinstance(row['topic_and_feelings_' + str(set+1)], float):
                t, f = row['topic_and_feelings_' + str(set+1)].split(' made me feel ')
                f = f.split(' ')
                f = [i.replace('.', '') for i in f]
                t_use.append([t.lower()])
                f_use.append(f)
        topics.append(t_use)
        emotions.append(f_use)

    # t_names = np.array([c.replace('Answer.t1.','') for c in df.columns if 't1' in c])
    # f_names = np.array([c.replace('Answer.f1.','') for c in df.columns if 'f1' in c])
    #
    # f_idx_set, t_idx_set = [], []
    # for set in ['1', '2', '3']:
    #     f_cols = [col for col in df.columns if 'f' + set in col]
    #     t_cols = [col for col in df.columns if 't' + set in col]
    #     f_idx = [df.columns.get_loc(c) for c in f_cols]
    #     t_idx = [df.columns.get_loc(c) for c in t_cols]
    #     f_idx_set.append(f_idx)
    #     t_idx_set.append(t_idx)
    #
    # for i, row in df.iterrows():
    #     f_use, t_use = [], []
    #     for set in range(3):
    #         t, f = np.array(row.values[t_idx_set[set]]).astype(bool), np.array(row.values[f_idx_set[set]]).astype(bool)
    #         t_true, f_true = t_names[t], f_names[f]
    #         t_use.append(list(t_true))
    #         f_use.append(list(f_true))
    #     topics.append(t_use)
    #     emotions.append(f_use)

    # topics = [['sleep'], ['work'], ['school']]
    # emotions = [['happy', 'satisfied'], ['anxious', 'afraid'], ['proud', 'calm']]

    # Lemotif adjustable settings
    args = {}
    # Algorithm for output
    args['algorithm'] = 'tiles'
    # Canvas size for output
    args['size'] = (500, 500)
    # Canvas background in BGR format
    args['background'] = (255, 255, 255)
    # Base size of icon relative to canvas size
    args['icon_ratio'] = 0.1
    # Standard deviation of a normal distribution centered at 1 from which icon resizing factors are sampled
    args['size_flux'] = 0.33
    # Number of times the canvas is iterated over; smaller numbers retain appearance of shapes, larger numbers appear more painterly
    args['passes'] = 8
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
    # Scalar representing the relative brightness value of borders
    args['border_color'] = .75

    args['out_dir'] = '/Users/alice/School/vil-lemotif/lemotif/output/tile'

    generate_visual(icons, colors, topics, emotions, **args)