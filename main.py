from flask import Flask, render_template, request

from app import utils
from lemotif import generator
from lemotif.visualizations.overlap import overlap


app = Flask(__name__, template_folder='app/templates', static_folder='app/static')


@app.route('/')
def home():
    subjects, emotions = generator.load_assets('assets')
    args, values = utils.set_args()
    return render_template('index.html',
                           emotions=emotions,
                           subjects=sorted(subjects.keys()),
                           images=None,
                           settings=args,
                           values=values)


@app.route('/', methods=['POST'])
def generate():
    subjects, emotions = generator.load_assets('assets')
    subjects_render, emotions_render = [], []
    for idx in range(4):
        subject_input = request.form.getlist('subjects' + str(idx+1))
        emotions_input = request.form.getlist('emotions' + str(idx+1))
        if len(subject_input[0]) > 0 and len(emotions_input[0]) > 0:
            subjects_render.append(subject_input)
            emotions_split = emotions_input[0].split(', ')
            emotions_split = emotions_split[:-1] if emotions_split[-1] == '' else emotions_split
            emotions_render.append(emotions_split)
    args, values = utils.get_args()

    motifs = generator.generate_visual(icons=subjects,
                                       colors=emotions,
                                       topics=subjects_render,
                                       emotions=emotions_render,
                                       algorithm=overlap,
                                       out=None,
                                       **args)
    images_encoded = []
    for motif in motifs:
        images_encoded.append(utils.img_to_str(motif))
    return render_template('index.html',
                           emotions=emotions,
                           subjects=sorted(subjects.keys()),
                           images=images_encoded,
                           settings=args,
                           values=values)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)