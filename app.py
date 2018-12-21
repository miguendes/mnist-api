import os

from flask import Flask, jsonify, request

import ml_models

app = Flask(__name__)


@app.route('/')
def index():
    """ The index endpoint. It just shows a message. """
    return 'Web API | MNIST Challenge'


@app.route('/predict/', methods=['POST'])
def predict():
    """ Predicts the digit corresponding to the image passed.

    This endpoint expects a JSON with the following structure:

    {
        "model": "model_name",
        "image": "base_64_string_that_represents_a_image"

    }

    """
    # TODO: perform JSON validatition. Look marshmallow project.
    data = request.get_json()
    model = data['model']
    image = data['image']

    model = ml_models.fetch(model)
    prediction = model.predict(image)

    return jsonify({"prediction": prediction})


@app.route('/models/')
def models():
    """ Returns a list of all models available. """
    return ml_models.list_models()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
