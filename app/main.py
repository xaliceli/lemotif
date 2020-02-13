import csv
import random
import string
from flask import Flask, render_template, request

import utils
from lemotif import generator
from lemotif.parsers import extract_labels

app = Flask(__name__, template_folder='templates', static_folder='static')
parser = extract_labels.BERTClassifier('models/bert', None, 29, batch_size=3)


@app.route('/', methods=['GET', 'POST'])
def home():
    subjects, emotions = generator.load_assets('static/images/icons')

    if request.method == 'GET':
        args, values = utils.set_args()
        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=None,
                               settings=args,
                               values=values,
                               error=None,
                               n_inputs=4)
    elif request.method == 'POST':
        all_text = []
        for idx in range(4):
            text_input = request.form.getlist('text' + str(idx + 1))
            all_text.append(text_input)
        subjects_render, emotions_render = parser.predict(all_text)
        # subject_input = request.form.getlist('subjects' + str(idx+1))
        # emotions_input = request.form.getlist('emotions' + str(idx+1))
        # if len(subject_input[0]) > 0 and len(emotions_input[0]) > 0:
        #     subjects_render.append(subject_input)
        #     emotions_split = emotions_input[0].split(', ')
        #     emotions_split = emotions_split[:-1] if emotions_split[-1] == '' else emotions_split
        #     emotions_render.append(emotions_split)
        args, values = utils.get_args()

        error, images_encoded = None, []
        try:
            motifs = generator.generate_visual(icons=subjects,
                                               colors=emotions,
                                               topics=subjects_render,
                                               emotions=emotions_render,
                                               out_dir=None,
                                               **args)

            for motif in motifs:
                images_encoded.append(utils.img_to_str(motif))
        except:
            error = 'Sorry, there was an error generating motifs for the provided inputs. This demo currently only supports emotions and topics in the dropdown lists. Please try again.'

        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=images_encoded,
                               settings=args,
                               values=values,
                               error=error,
                               emot_labels=emotions_render,
                               subj_labels=subjects_render,
                               n_inputs=4)


@app.route('/demo/', methods=['GET', 'POST'])
def demo():
    subjects, emotions = generator.load_assets('static/images/icons')

    if request.method == 'GET':
        args, values = utils.set_args()
        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=None,
                               settings=args,
                               values=values,
                               error=None,
                               demo=True,
                               n_inputs=3)

    elif request.method == 'POST':
        all_text = []
        for idx in range(3):
            text_input = request.form.getlist('text' + str(idx + 1))
            all_text.append(text_input)
        subjects_render, emotions_render = parser.predict(all_text)
        args, values = utils.get_args()

        error, images_encoded = None, []
        all_styles = ['carpet', 'circle', 'overlap', 'string', 'tiles', 'watercolors']
        random.shuffle(all_styles)
        try:
            motifs = generator.generate_visual(icons=subjects,
                                               colors=emotions,
                                               topics=subjects_render,
                                               emotions=emotions_render,
                                               out_dir=None,
                                               all_styles=all_styles,
                                               **args)

            for motif in motifs:
                images_encoded.append(utils.img_to_str(motif))
        except:
            error = 'Sorry, there was an error generating motifs for the provided inputs.'

        letters = string.ascii_lowercase
        completion_code = ''.join(random.choice(letters) for i in range(17))

        with open("eval_logs.csv", "a") as log:
            row = csv.writer(log)
            row.writerow([completion_code] + all_styles)

        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=images_encoded,
                               settings=args,
                               values=values,
                               error=error,
                               emot_labels=emotions_render,
                               subj_labels=subjects_render,
                               demo=True,
                               n_inputs=3,
                               completion_code=completion_code)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)