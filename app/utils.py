"""
utils.py
Helpers.
"""

from flask import request
from urllib.parse import quote
from PIL import Image
from base64 import b64encode
from io import BytesIO


def set_args():
    """Default arguments for app functions."""
    args = {}

    # Show settings pane
    args['settings_display'] = 'none'
    # Algorithm to use
    args['algorithm'] = 'watercolors'
    # Canvas size for output
    args['size'] = (256, 256)
    # Canvas background in BGR format
    args['background'] = (255, 255, 255)

    # CARPET ARGS
    args['tile_ratio'] = 0.1
    args['line_width'] = 1
    args['rot_degree'] = 45

    # CIRCLE ARGS
    args['n_circles'] = 100
    args['min_rad_factor'] = 0.01
    args['max_rad_factor'] = 0.10

    # GLASS ARGS
    args['icon_ratio'] = 0.1
    args['size_flux'] = 0.33
    args['passes'] = 5
    args['inc_floor'] = 0.5
    args['inc_ceiling'] = 0.75
    args['rand_alpha'] = True
    args['mask_all'] = True

    # TILE ARGS
    args['line_width_tile'] = 1
    args['dir_prob'] = .5
    args['step_size'] = 10

    # STRING ARGS
    args['n_lines'] = 150
    args['line_width_string'] = 5
    args['offset_sd'] = .2

    # WATERCOLORS ARGS
    args['intensity_sd'] = .2
    args['model_dir'] = 'models/ae'

    # If True, randomly select topic for border shape as opposed to using square.
    args['border_shape'] = True
    # Scalar representing the relative brightness value of borders
    args['border_color'] = 0.5
    # Display text labels at bottom
    args['text'] = True
    # Summary motif
    args['summary'] = False


    values = {
        'text1': '',
        'text2': '',
        'text3': '',
        'text4': ''
    }

    return args, values


def get_args():
    """Get arguments from form inputs and process to expected format."""
    args, values = set_args()
    int_params = ['rot_degree', 'line_width', 'line_width_tile', 'n_circles', 'passes', 'step_size', 'n_lines', 'line_width_string']

    for param in args.keys():
        results = request.form.getlist(param)
        if len(results) > 0:
            val = results[0]
            if type(args[param]) is bool:
                val = val == 'True'
            elif type(args[param]) is not str:
                val = float(val) / 100 if float(val) >= 1 >= args[param] and param not in int_params else int(val)
            args[param] = val

    for value in values.keys():
        results = request.form.getlist(value)
        if len(results) > 0:
            results[0] = results[0][:-2] if results[0][-2:] == ', ' else results[0]
            values[value] = results[0]

    return args, values


def img_to_str(img, web=True):
    """Format image array into byte string."""
    image = Image.fromarray(img.astype("uint8")[..., [2, 1, 0]])
    rawBytes = BytesIO()
    image.save(rawBytes, 'PNG')
    rawBytes.seek(0)
    data = b64encode(rawBytes.read())
    data_url = 'data:image/png;base64,{}'.format(quote(data)) if web else data.decode()
    return data_url