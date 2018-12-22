import json
import unittest
from os import path
from unittest import mock

import pytest

from app import app

path_prefix = path.dirname(path.abspath(__file__))
fixtures_path = path.join(path_prefix, 'fixtures')


def ok_three_file():
    return path.join(fixtures_path, 'ok_three.base64')


def large_four_file():
    return path.join(fixtures_path, 'large_four.base64')


class TestMNISTApp(unittest.TestCase):
    """Class to test the flask app endpoints"""

    @classmethod
    def setUpClass(cls):
        app.config['SERVER_NAME'] = 'localhost:5000'
        cls.client = app.test_client()

    def setUp(self):
        """Set up application for testing."""
        self.app_context = app.app_context()
        self.app_context.push()

    def test_home(self):
        """Tests if accessing the index the welcome message is the same as defined in the endpoint."""
        response = self.client.get('/')
        self.assertEqual(200, response.status_code)
        self.assertIn('Web API | MNIST Challenge', response.data.decode('utf-8'))

    @mock.patch('app.ml_models.fetch')
    def test_predict_with_image_from_dataset(self, fetch_model_mock):
        """Tests if given a valid JSON, the prediction is returned. """
        model_name = 'svm'
        image_b64 = ('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                     'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                     'AAAAAAAAAAAAAAADEhISfoivGqb/938AAAAAAAAAAAAAAAAeJF6aqv39/f394az98sNAAAAAAAAAAAAAAAAx7v39/f39/f3'
                     '9+11SUjgnAAAAAAAAAAAAAAAAEtv9/f39/ca29/EAAAAAAAAAAAAAAAAAAAAAAABQnGv9/c0LACuaAAAAAAAAAAAAAAAAAA'
                     'AAAAAAAA4Bmv1aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIv9vgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALvv1GAAAAA'
                     'AAAAAAAAAAAAAAAAAAAAAAAAAAAACPx4aBsAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUfD9/XcZAAAAAAAAAAAAAAAAAAAA'
                     'AAAAAAAAAAAtuv39lhsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBd/P27AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPn9+UA'
                     'AAAAAAAAAAAAAAAAAAAAAAAAAAAAugrf9/c8CAAAAAAAAAAAAAAAAAAAAAAAAACeU5f39/fq2AAAAAAAAAAAAAAAAAAAAAA'
                     'AAGHLd/f39/clOAAAAAAAAAAAAAAAAAAAAAAAXQtX9/f39xlECAAAAAAAAAAAAAAAAAAAAABKr2/39/f3DUAkAAAAAAAAAA'
                     'AAAAAAAAAAAN6zi/f39/fSFCwAAAAAAAAAAAAAAAAAAAAAAAIj9/f3Uh4QQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                     'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                     'AAA==')

        model_mock = mock.Mock()
        model_mock.predict.return_value = 9

        fetch_model_mock.return_value = model_mock

        response = self.client.post('/predict/', json={'model': model_name, 'image': image_b64})

        fetch_model_mock.assert_called_once_with(model_name)
        model_mock.predict.assert_called_once_with(image_b64)

        self.assertEqual(200, response.status_code)
        self.assertEqual({"prediction": 9}, json.loads(response.get_data(as_text=True)))

    def test_predict_with_real_model(self):
        """Tests if given a valid JSON, the prediction is returned. """
        model_name = 'cnn'
        with open(large_four_file()) as img:
            image_b64 = img.read().replace('\n', '')

        response = self.client.post('/predict/', json={'model': model_name, 'image': image_b64})

        self.assertEqual(200, response.status_code)
        self.assertEqual({"prediction": 4}, json.loads(response.get_data(as_text=True)))

    def test_list_models(self):
        response = self.client.get('/models/')
        self.assertEqual(200, response.status_code)
        self.assertEqual({"models": ["cnn", "mlp", "svm"]}, json.loads(response.get_data(as_text=True)))


    def tearDown(self):
        """Tear down method to get rid of flask context created."""
        self.app_context.pop()


if __name__ == '__main__':
    unittest.main()
