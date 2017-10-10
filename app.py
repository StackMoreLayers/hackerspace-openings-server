from flask import Flask

from . import yolo

app = Flask(__name__)

MODELS = [
    yolo.Yolo
]


@app.route("/")
def hello():
    return "Hello World!"
