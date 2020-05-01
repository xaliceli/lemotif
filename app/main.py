from flask import Flask, render_template, request, session
import numpy as np

import utils
from config import Config
from lemotif import generator
from lemotif.parsers import extract_labels

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config())

parser = extract_labels.BERTClassifier('models/bert', None, 29, batch_size=3)

@app.route('/', methods=['GET', 'POST'])
def home():
    subjects, emotions = generator.load_assets('static/images/icons')

    if request.method == 'GET':
        session['text'] = []
        args, values = utils.set_args()
        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=None,
                               combined=None,
                               settings=args,
                               values=values,
                               error=None,
                               n_inputs=3)
    elif request.method == 'POST':
        all_text = []
        for idx in range(3):
            text_input = request.form.getlist('text' + str(idx + 1))
            all_text.append(text_input)

        if np.all(np.logical_or(np.array(all_text) == np.array(session['text']), np.array(all_text) == '')):
            subjects_render = session['subjects']
            emotions_render = session['emotions']
        else:
            subjects_render, emotions_render = parser.predict(all_text)
            session['text'] = all_text
            session['subjects'] = subjects_render
            session['emotions'] = emotions_render
        subjects_render = [x if all_text[i][0] != '' else [''] for i, x in enumerate(subjects_render)]

        args, values = utils.get_args()

        error, images_encoded = None, []
        try:
            motifs, combined = generator.generate_visual(icons=subjects,
                                                         colors=emotions,
                                                         topics=subjects_render,
                                                         emotions=emotions_render,
                                                         out_dir=None,
                                                         **args)
            for motif in motifs:
                if motif is None:
                    images_encoded.append('')
                else:
                    images_encoded.append(utils.img_to_str(motif[:-20, ...]))
            combined_encoded = utils.img_to_str(combined)
        except:
            error = 'Sorry, there was an error generating motifs for the provided inputs. Please try again.'

        subjects_render = [['topic unknown'] if x[0] is None else x for x in subjects_render]
        subjects_render = [x for i, x in enumerate(subjects_render) if all_text[i][0] != '']
        emotions_render = [x for i, x in enumerate(emotions_render) if all_text[i][0] != '']

        return render_template('index.html',
                               emotions=emotions,
                               subjects=sorted(subjects.keys()),
                               images=images_encoded,
                               combined=combined_encoded,
                               settings=args,
                               values=values,
                               error=error,
                               emot_labels=emotions_render,
                               subj_labels=subjects_render,
                               n_inputs=3)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8081, debug=True)