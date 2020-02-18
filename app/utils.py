"""
utils.py
Helpers
"""

from flask import request
from urllib.parse import quote
from PIL import Image
from base64 import b64encode
from io import BytesIO


def set_args():
    args = {}
    # Algorithm to use
    args['algorithm'] = 'watercolors'
    # Canvas size for output
    args['size'] = (256, 256)
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
    # Scalar representing the relative brightness value of borders
    args['border_color'] = 0.5
    # Display text labels at bottom
    args['text'] = False
    # Summary motif
    args['summary'] = False

    # AE specific args
    args['model_dir'] = 'models/ae'

    values = {
        'text1': '',
        'text2': '',
        'text3': '',
        'text4': ''
    }

    return args, values

def get_args():
    args, values = set_args()

    for param in args.keys():
        results = request.form.getlist(param)
        if len(results) > 0:
            val = results[0]
            if type(args[param]) is bool:
                val = val == 'True'
            elif type(args[param]) is not str:
                val = float(val) / 100 if float(val) >= 1 >= args[param] and param != 'passes' else int(val)
            args[param] = val

    for value in values.keys():
        results = request.form.getlist(value)
        if len(results) > 0:
            results[0] = results[0][:-2] if results[0][-2:] == ', ' else results[0]
            values[value] = results[0]

    return args, values

def img_to_str(img):
    image = Image.fromarray(img.astype("uint8")[..., [2, 1, 0]])
    rawBytes = BytesIO()
    image.save(rawBytes, 'PNG')
    rawBytes.seek(0)
    data = b64encode(rawBytes.read())
    data_url = 'data:image/png;base64,{}'.format(quote(data))
    return data_url