import binascii
import os

from decouple import config
from flask import Flask, jsonify, request, render_template
from flask_caching import Cache

import ml_models

app = Flask(__name__)

cache = Cache(app, config={
    'CACHE_TYPE': config('CACHE_TYPE', default='null'),
    'CACHE_KEY_PREFIX': config('CACHE_KEY_PREFIX', default='fcache'),
    'CACHE_REDIS_URL': config('CACHE_REDIS_URL', default='redis://localhost:6379')
})


@app.route('/')
def index():
    """ The index endpoint. It renders the home page. """
    return render_template("home.html")


@app.route('/predict/', methods=['POST'])
@cache.cached(timeout=50)
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

    try:
        model = ml_models.fetch(model)
        prediction = model.predict(image)
    except ml_models.ModelNotFoundError:
        available_models = ', '.join(ml_models.list_models())
        return jsonify({"error": f"Unexpected model name. Only {available_models} models are available."}), 422
    except (OSError, TypeError, binascii.Error):
        return jsonify({"error": "Could not perform the prediction. Invalid image base64 image."}), 422
    return jsonify({"prediction": prediction}), 200


@app.route('/models/')
@cache.cached(timeout=50)
def models():
    """ Returns a list of all models available. """
    return jsonify({"models": ml_models.list_models()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
