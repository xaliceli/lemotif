from flask import Flask, render_template, request
from lemotif import generator


app = Flask(__name__)


@app.route('/')
def home():
    subjects, emotions = generator.load_assets('assets')
    return render_template('index.html',
                           emotions=sorted(emotions.keys()),
                           subjects=sorted(subjects.keys()),
                           image=None)


@app.route('/', methods=['POST'])
def search():
    return NotImplementedError

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)