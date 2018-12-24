import json
from os import path

import pytest

from app import app

path_prefix = path.dirname(path.abspath(__file__))
fixtures_path = path.join(path_prefix, 'fixtures')
ok_three_file = path.join(fixtures_path, 'ok_three.base64')
large_four_file = path.join(fixtures_path, 'large_four.base64')


@pytest.fixture
def client(request):
    return app.test_client()


def test_home(client):
    """Tests if accessing the index the welcome message is the same as defined in the endpoint."""
    response = client.get('/')
    assert 200 == response.status_code
    assert 'Web API | MNIST Challenge' in response.data.decode('utf-8')
    assert ('A Flask REST API for handwritten digit recognition'
            ' using machine learning models.') in response.data.decode('utf-8')


def test_list_models(client):
    response = client.get('/models/')
    assert 200 == response.status_code
    assert {"models": ["cnn", "mlp", "svm"]} == json.loads(response.get_data(as_text=True))


@pytest.mark.parametrize("model_name", [
    "svm",
    "cnn",
    "mlp",
])
@pytest.mark.parametrize("base64_image_file, expected_prediction", [
    (ok_three_file, 3),
    (large_four_file, 4),
])
def test_predict_with_real_model(client, base64_image_file, expected_prediction, model_name):
    """Tests if given a valid JSON, the prediction is returned. """
    with open(base64_image_file) as img:
        image_b64 = img.read().replace('\n', '')

    response = client.post('/predict/', json={'model': model_name, 'image': image_b64})

    assert 200 == response.status_code
    assert {"prediction": expected_prediction} == json.loads(response.get_data(as_text=True))


def test_unavailable_model(client):
    """Tests if an error message is returned when a model does not exist."""
    available_models = ', '.join(["cnn", "mlp", "svm"])
    response = client.post('/predict/', json={'model': 'unavailable_model', 'image': 'image_b64'})
    assert 422 == response.status_code
    assert ({"error": f"Unexpected model name. Only {available_models} models are available."}
            == json.loads(response.get_data(as_text=True)))
