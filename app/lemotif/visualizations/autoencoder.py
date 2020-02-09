"""
autoencoder.py
WikiArt trained autoencoder
"""
from itertools import product

import numpy as np
import os
import tensorflow as tf

from lemotif.visualizations.utils import fill_color, bg_mask, fill_canvas, apply_shape, add_labels

def ae(topics, emotions, icons, colors, size, in_size=16, background=(255, 255, 255), model_dir='',
       border_shape=False, border_color=None, text=True, **kwargs):
    if not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    if not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'

    colors_list = [colors[emotion]['rgb'] for emotion in emotions]
    canvas = np.ones((in_size, in_size, 3))
    coords = np.reshape(np.indices((in_size, in_size)).transpose((1, 2, 0)), (in_size * in_size, 2))
    np.random.shuffle(coords)
    coords_split = np.array_split(coords, len(colors_list))
    for idx, c in enumerate(colors_list):
        canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]] = c[::-1]
        canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]] *= np.random.normal(loc=1, scale=.2, size=(
            canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]].shape))

    img_proc = np.asarray(canvas, dtype='float32')[None, ...]
    img_proc = (img_proc / 127.5 - 1.0).astype(np.float32)

    enc_model = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.h5'))
    dec_model = tf.keras.models.load_model(os.path.join(model_dir, 'decoder.h5'))

    img_out = dec_model.predict(enc_model.predict(img_proc, batch_size=1), batch_size=1)
    img_out = (img_out[0, ..., :3] * 127.5 + 127.5).astype(np.uint8)

    # base_size = int(min(size) * icon_ratio)
    # icons_resized = [cv2.resize(icons[topic], (base_size, base_size)) for topic in topics]
    # colors_list = [colors[emotion] for emotion in emotions]
    # color_icons = [fill_color(icon, color['rgb'], border_color) for icon, color in product(icons_resized, colors_list)]
    #
    # canvas = np.zeros((size[0], size[1], 3))
    # canvas[..., :] = background
    # for p in range(passes):
    #     complete = False
    #     start = [random.randint(0, base_size // 4), random.randint(0, base_size // 4)]
    #     while not complete:
    #         icon, icon_size = random.choice(color_icons), max(1, int(base_size * np.random.normal(1, size_flux)))
    #         icon_resized = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_NEAREST)
    #         adj_y = min(int(start[0] * np.random.normal(1, .025)), size[0] - icon_size)
    #         mask = bg_mask(icon_resized, colors_list, border_color) if mask_all else None
    #         increment = random.randint(int(inc_floor * icon_size), int(icon_size * inc_ceiling))
    #         alpha = random.random() if rand_alpha else 0
    #         if start[1] + icon_size < canvas.shape[1]:
    #             canvas = fill_canvas(canvas, background, mask, size, icon_resized, start, adj_y, icon_size, alpha)
    #             start[1] += increment
    #         elif start[0] + icon_size < canvas.shape[0]:
    #             start[0] += increment
    #             start[1] = increment
    #             canvas = fill_canvas(canvas, background, mask, size, icon_resized, start, adj_y, icon_size, alpha)
    #         else:
    #             complete = True

    if border_shape:
        img_out = apply_shape(img_out, icons, topics, size, border_color, background)

    if text:
        img_out = add_labels(img_out, topics, emotions, colors)

    return img_out