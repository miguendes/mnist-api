from flask import Flask, jsonify, request

import ml_models

app = Flask(__name__)


@app.route('/')
def index():
    return 'Web API | MNIST Challenge'


if __name__ == "__main__":
    app.run()
