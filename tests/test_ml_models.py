import unittest
from os import path
from unittest.mock import patch

import ml_models

path_prefix = path.dirname(path.abspath(__file__))
fixtures_path = path.join(path_prefix, 'fixtures')
ok_three_file = path.join(fixtures_path, 'ok_three.base64')
large_four_file = path.join(fixtures_path, 'large_four.base64')


@patch('ml_models.jsonload', return_value={
    "svm": ["tests", "weights", "svm_v1.pkl"],
    "cnn": ["tests", "weights", "cnn_v_test.h5"],
    "mlp": ["tests", "weights", "mlp_v1.h5"],
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
        with self.assertRaises(ml_models.ModelNotFoundError) as context:
            name = 'non_existent_model'
            ml_models.fetch(name)
        self.assertTrue(f'Model named {name} does not exist.' in str(context.exception))

    def test_predict_with_image_with_28by28_dimension_on_svm_model(self, mock):
        """Tests if the SVM can predict a image with correct dimension of 28x28. """
        model_name = 'svm'
        with open(ok_three_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)
        self.assertEqual(3, prediction)

    def test_predict_with_larger_image_on_svm_model(self, mock):
        """Tests if given a valid JSON, the prediction is returned. """
        model_name = 'svm'
        with open(large_four_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)

        self.assertEqual(4, prediction)

    def test_predict_with_image_with_28by28_dimension_on_cnn_model(self, mock):
        """Tests if the SVM can predict a image with correct dimension of 28x28. """
        model_name = 'cnn'
        with open(ok_three_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)
        self.assertEqual(3, prediction)

    def test_predict_with_larger_image_on_cnn_model(self, mock):
        """Tests if given a valid JSON, the prediction is returned. """
        model_name = 'cnn'
        with open(large_four_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)

        self.assertEqual(4, prediction)

    def test_predict_with_image_with_28by28_dimension_on_mlp_model(self, mock):
        """Tests if the SVM can predict a image with correct dimension of 28x28. """
        model_name = 'cnn'
        with open(ok_three_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)
        self.assertEqual(3, prediction)

    def test_predict_with_larger_image_on_mlp_model(self, mock):
        """Tests if given a valid JSON, the prediction is returned. """
        model_name = 'cnn'
        with open(large_four_file) as img:
            image_b64 = img.read().replace('\n', '')

        model = ml_models.fetch(model_name)

        prediction = model.predict(image_b64)

        self.assertEqual(4, prediction)


if __name__ == '__main__':
    unittest.main()
