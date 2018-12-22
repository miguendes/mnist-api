from os import path
from unittest.mock import patch

import pytest

import ml_models

path_prefix = path.dirname(path.abspath(__file__))
fixtures_path = path.join(path_prefix, 'fixtures')
ok_three_file = path.join(fixtures_path, 'ok_three.base64')
large_four_file = path.join(fixtures_path, 'large_four.base64')


def test_list_models_available():
    """Tests if the available models is the same as the ones defined in the config file."""
    assert ['cnn', 'mlp', 'svm'] == ml_models.list_models()


@pytest.mark.parametrize("model_name, model_instance", [
    ('cnn', ml_models.CNNModel),
    ('mlp', ml_models.MLPModel),
    ('svm', ml_models.SVMModel),
])
def test_fetch_available_models(model_name, model_instance):
    """Tests if the model returned is the same as the one passed as param."""
    model = ml_models.fetch(model_name)
    assert isinstance(model, model_instance)


def test_unavailable_model():
    """Tests if the function raises an exception if it can't find a model."""
    with pytest.raises(ml_models.ModelNotFoundError) as context:
        model = 'non_existent_model'
        ml_models.fetch(model)
    assert f'Model named {model} does not exist.' in str(context.value)


@patch('ml_models.jsonload', return_value={
    "svm": ["tests", "weights", "svm_v1.pkl"],
    "cnn": ["tests", "weights", "cnn_v_test.h5"],
    "mlp": ["tests", "weights", "mlp_v1.h5"],
})
class TestMNISTModelsPrediction:
    @pytest.mark.parametrize("model_name", [
        "svm",
        "cnn",
        "mlp",
    ])
    @pytest.mark.parametrize("base64_image_file, expected_prediction", [
        (ok_three_file, 3),
        (large_four_file, 4),
    ])
    def test_predict_image(self, mock, model_name, base64_image_file,
                                                                   expected_prediction):
        """Tests if the model can predict a image with correct dimension of 28x28. """
        print(mock, base64_image_file, expected_prediction)

        with open(base64_image_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)
        assert expected_prediction == prediction
