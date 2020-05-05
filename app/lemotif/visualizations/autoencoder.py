"""
autoencoder.py
WikiArt trained autoencoder.
"""

import numpy as np
import tensorflow as tf

from lemotif.visualizations.utils import fill_color, bg_mask, fill_canvas, apply_shape, add_labels

session = tf.Session()

with session.as_default():
    with session.graph.as_default():
        enc_model = tf.keras.models.load_model('models/ae/encoder.h5')
        dec_model = tf.keras.models.load_model('models/ae/decoder.h5')
        enc_model._make_predict_function()
        dec_model._make_predict_function()

def ae(topics, emotions, icons, colors, size, in_size=16, background=(255, 255, 255),
       border_shape=False, border_color=None, text=True, intensity_sd=.2, **kwargs):
    """
    Autoencoder visualization.

    :param topics: Topics to use (list).
    :param emotions: Emotions to use (list).
    :param icons: Shape icons (dict).
    :param colors: Emotion colors (dict).
    :param size: Size of output (tuple).
    :param in_size: Size of input, assuming square shape (int).
    :param background: Background color in RGB (tuple).
    :param border_shape: Whether to apply icon shape as border (bool).
    :param border_color: Border color in RGB (tuple).
    :param text: Include text labels below visualization (bool).
    :param intensity_sd: Standard deviation for random color intensity adjustment (float).
    :param kwargs: Additional arguments (dict).
    :return: Visualization (array).
    """
    if len(topics) == 0 or len(emotions) == 0:
        return None
    elif topics[0] is None:
        border_shape = False
    elif not set(topics) <= set(icons.keys()):
        return 'Error: Topics outside of presets.'
    elif not set(emotions) <= set(colors.keys()):
        return 'Error: Emotions outside of presets.'

    colors_list = [colors[emotion]['rgb'] for emotion in emotions]
    canvas = np.ones((in_size, in_size, 3))
    coords = np.reshape(np.indices((in_size, in_size)).transpose((1, 2, 0)), (in_size * in_size, 2))
    np.random.shuffle(coords)
    coords_split = np.array_split(coords, len(colors_list))
    for idx, c in enumerate(colors_list):
        canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]] = c[::-1]
        canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]] *= np.random.normal(loc=1, scale=intensity_sd, size=(
            canvas[coords_split[idx][:, 0], coords_split[idx][:, 1]].shape))

    img_proc = np.asarray(canvas, dtype='float32')[None, ...]
    img_proc = (img_proc / 127.5 - 1.0).astype(np.float32)

    with session.as_default():
        with session.graph.as_default():
            img_out = dec_model.predict(enc_model.predict(img_proc, batch_size=1), batch_size=1)
    img_out = (img_out[0, ..., :3] * 127.5 + 127.5).astype(np.uint8)

    if border_shape:
        img_out = apply_shape(img_out, icons, topics, size, border_color, background)

    if text:
        img_out = add_labels(img_out, topics, emotions, colors)

    return img_out