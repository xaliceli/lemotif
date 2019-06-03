from flask import Flask, render_template, request, redirect
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
    subjects, emotions = generator.load_assets('assets')
    subjects_render = request.form.getlist('subjects')
    emotions_render = request.form.getlist('emotions')
    print(subjects_render, emotions_render)
    lemotif = generator.generate_visual(icons=subjects,
                                        colors=emotions,
                                        topics=[subjects_render],
                                        emotions=[emotions_render],
                                        algorithm=overlap,
                                        out=None,
                                        size=(500, 500))[0]
    image = Image.fromarray(lemotif.astype("uint8"))
    rawBytes = BytesIO()
    image.save(rawBytes, 'PNG')
    rawBytes.seek(0)
    data = b64encode(rawBytes.read())
    data_url = 'data:image/png;base64,{}'.format(quote(data))
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=data_url)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)