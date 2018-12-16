import unittest
from unittest.mock import patch

import ml_models


@patch('ml_models.jsonload', return_value={
    "svm": "tests/weights/svm_v1.pkl",
    "cnn": "tests/weights/cnn_v_test.h5",
    "mlp": "tests/weights/mlp_v1.h5",
})
class TestMNISTModels(unittest.TestCase):
    """A class that contains the tests for the module ml_models."""

    def test_list_models_available(self, mock):
        """Tests if the available models is the same as the ones defined in the config file."""
        self.assertListEqual(['cnn', 'mlp', 'svm'], ml_models.list_models())

    def test_fetch_available_models(self, mock):
        """Tests if the model returned is the same as the one passed as param."""
        svm_model = ml_models.fetch('svm')
        self.assertIsInstance(svm_model, ml_models.SVMModel)

        cnn_model = ml_models.fetch('cnn')
        self.assertIsInstance(cnn_model, ml_models.CNNModel)

        mlp_model = ml_models.fetch('mlp')
        self.assertIsInstance(mlp_model, ml_models.MLPModel)

    def test_unavailable_model(self, mock):
        """Tests if the function raises an exception if it can't find a model."""
        name = 'non_existent_model'
        with self.assertRaises(ml_models.ModelNotFoundError) as context:
            ml_models.fetch(name)
            self.assertTrue('Model named {name} does not exist.' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
