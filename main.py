from flask import Flask, render_template, request
from urllib.parse import quote
from PIL import Image
from base64 import b64encode
from io import BytesIO

from lemotif import generator
from lemotif.visualizations.overlap import overlap

app = Flask(__name__)


@app.route('/')
def home():
    subjects, emotions = generator.load_assets('assets')
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=None)


@app.route('/generate', methods=['POST'])
def generate():
    def get_args():
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

        for param in args.keys():
            results = request.form.getlist(param)
            if len(results) > 0:
                val = results[0]
                if type(args[param]) is bool:
                    val = val == 'True'
                else:
                    val = float(val)/100 if float(val) > 1 >= args[param] else int(val)
                args[param] = val
        if request.form.getlist('border_color_toggle')[0] == 'False':
            args['border_color'] = None

        return args

    def img_to_str(img):
        image = Image.fromarray(img.astype("uint8"))
        rawBytes = BytesIO()
        image.save(rawBytes, 'PNG')
        rawBytes.seek(0)
        data = b64encode(rawBytes.read())
        data_url = 'data:image/png;base64,{}'.format(quote(data))
        return data_url

    subjects, emotions = generator.load_assets('assets')
    subjects_render = request.form.getlist('subjects')
    emotions_render = request.form.getlist('emotions')
    args = get_args()

    lemotif = generator.generate_visual(icons=subjects,
                                        colors=emotions,
                                        topics=[subjects_render],
                                        emotions=[emotions_render],
                                        algorithm=overlap,
                                        out=None,
                                        **args)[0]
    image_encoded = img_to_str(lemotif)
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=image_encoded)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)