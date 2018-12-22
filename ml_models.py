import base64
import io
from json import load as jsonload
from os import path

import cv2
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model as load
from sklearn.externals import joblib

path_prefix = path.dirname(path.abspath(__file__))
config_path = path.join(path_prefix, 'config.json')


def load_weights():
    """Loads models weights from config file."""
    with open(config_path) as f:
        return jsonload(f)


class ModelNotFoundError(Exception):
    """Custom exception to indicate when a model does not exist or could not be found. """

    def __init__(self, message):
        super().__init__(message)


def fetch(name):
    """
    Fetches an appropriate model to perform the prediction.
    :param name: model's name
    :return: a trained model
    """
    K.clear_session()

    try:
        full_weights_path = path.join(path_prefix, *load_weights()[name])

        if name == 'svm':
            return SVMModel(joblib.load(full_weights_path))
        elif name == 'cnn':
            return CNNModel(load(full_weights_path))
        elif name == 'mlp':
            return MLPModel(load(full_weights_path))
    except KeyError:
        raise ModelNotFoundError(f'Model named {name} does not exist.')


def _b64_to_image(base64_string):
    """
    Private function to convert a base64 string into a Numpy array.
    :param base64_string:
    :return: Numpy array
    """
    imgdata = base64.b64decode(str(base64_string))
    output = io.BytesIO(imgdata)
    output.seek(0)
    image = Image.open(output)
    original_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return cv2.resize(original_image, (28, 28), cv2.INTER_AREA)


class CNNModel:
    """
    A class wrapper for a Convolutional Neural Network implemented in Keras.

    """

    def __init__(self, model):
        self.model = model

    def predict(self, image_b64):
        """
        Predicts the label of the image.

        :param image_b64:
        :return: prediction as integer
        """
        image = _b64_to_image(image_b64)
        return self.model.predict_classes(image.reshape(-1, 28, 28, 1)).tolist()[0]


class MLPModel:
    """
    A class wrapper for a Multilayer Perceptron implemented in Keras.

    """

    def __init__(self, model):
        self.model = model

    def predict(self, image_b64):
        """
        Predicts the label of the image.

        :param image_b64:
        :return: prediction as integer
        """
        image = _b64_to_image(image_b64)
        return self.model.predict_classes(image.reshape(-1, 28 * 28)).tolist()[0]


class SVMModel:
    """
    A class wrapper for a Support Vector Machine implemented in scikit-learn.

    """

    def __init__(self, model):
        self.model = model

    def predict(self, image_b64):
        """
        Predicts the label of the image.

        :param image_b64:
        :return: prediction as integer
        """
        image = _b64_to_image(image_b64)
        return self.model.predict([image.reshape(-1)]).tolist()[0]


def list_models():
    """Lists all available models."""
    return sorted(load_weights().keys())
