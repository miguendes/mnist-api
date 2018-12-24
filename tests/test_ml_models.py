import pytest

import ml_models


@pytest.mark.parametrize("model_name, model_instance", [
    ('cnn', ml_models.CNNModel),
    ('mlp', ml_models.MLPModel),
    ('svm', ml_models.SVMModel),
])
def test_fetch_available_models(model_name, model_instance):
    """Tests if the model returned is the same as the one passed as param."""
    model = ml_models.fetch(model_name)
    assert isinstance(model, model_instance)
