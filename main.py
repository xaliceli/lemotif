from flask import Flask, render_template, request

from app import utils
from lemotif import generator
from lemotif.visualizations.overlap import overlap


app = Flask(__name__, template_folder='app/templates/')


@app.route('/')
def home():
    subjects, emotions = generator.load_assets('assets')
    args = utils.set_args()
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=None,
                           settings=args)


@app.route('/', methods=['POST'])
def generate():
    subjects, emotions = generator.load_assets('assets')
    subjects_render = request.form.getlist('subjects')
    emotions_render = request.form.getlist('emotions')
    args = utils.get_args()

    lemotif = generator.generate_visual(icons=subjects,
                                        colors=emotions,
                                        topics=[subjects_render],
                                        emotions=[emotions_render],
                                        algorithm=overlap,
                                        out=None,
                                        **args)[0]
    image_encoded = utils.img_to_str(lemotif)
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=image_encoded,
                           settings=args)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)