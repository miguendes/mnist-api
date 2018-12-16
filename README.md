MNIST Api
==========================

[![Build Status](https://travis-ci.org/mendesmiguel/mnist-api.svg?branch=master)](https://travis-ci.org/mendesmiguel/mnist-api.svg?branch=master) 
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


A Flask REST API for handwritten digit recognition using machine learning models.

The goal of this project is to develop a REST Api that accepts an image and returns a
prediction.


Dependencies
------------
To run the solution you need the following dependencies:

- Python 3.6+
- Numpy 1.15+
- Flask 1.0.2
- Tensorflow 1.4
- Keras 2.0.8
- scikit-learn 0.20+
- Pipenv 2018+
- Jupyter Notebook
- matplotlib

To install Pipenv and Python 3.6 without messing up your environment in case you don't have them installed,
 I strongly suggest 
installing pyenv. You can follow 
[this amazing tutorial](https://medium.com/@henriquebastos/the-definitive-guide-to-setup-my-python-workspace-628d68552e14)
 from [@henriquebastos](https://github.com/henriquebastos).

Installation
------------

#### 1. Clone this repo

``` {.sourceCode .bash}
$ git clone git@github.com:mendesmiguel/mnist-api.git
```
#### 2. Install Dependencies

To install all dependencies, you can use [pipenv](http://pipenv.org/).

Pipenv will spin up a virtualenv and install the dependencies based on a `Pipenv.lock` file inside the root of
the project.

``` {.sourceCode .bash}
$ cd mnist-api/
$ pipenv install 
```

Usage
------------

To run the project locally do:
``` {.sourceCode .bash}
$ cd mnist-api/
$ pipenv shell
$ python app.py
```
The dev server will spin up and will be available at: http://127.0.0.1:5000/

Example of prediction:
``` {.sourceCode .bash}
$ curl -d '{"model":"cnn", "image": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADEhISfoivGqb/938AAAAAAAAAAAAAAAAeJF6aqv39/f394az98sNAAAAAAAAAAAAAAAAx7v39/f39/f39+11SUjgnAAAAAAAAAAAAAAAAEtv9/f39/ca29/EAAAAAAAAAAAAAAAAAAAAAAABQnGv9/c0LACuaAAAAAAAAAAAAAAAAAAAAAAAAAA4Bmv1aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIv9vgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALvv1GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACPx4aBsAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUfD9/XcZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAtuv39lhsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBd/P27AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPn9+UAAAAAAAAAAAAAAAAAAAAAAAAAAAAAugrf9/c8CAAAAAAAAAAAAAAAAAAAAAAAAACeU5f39/fq2AAAAAAAAAAAAAAAAAAAAAAAAGHLd/f39/clOAAAAAAAAAAAAAAAAAAAAAAAXQtX9/f39xlECAAAAAAAAAAAAAAAAAAAAABKr2/39/f3DUAkAAAAAAAAAAAAAAAAAAAAAN6zi/f39/fSFCwAAAAAAAAAAAAAAAAAAAAAAAIj9/f3Uh4QQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predict/
```

Methodology
------------

The task has an important restriction: only 500 images to train a model on, including the test set.
Since this amount is so little to train a good classifier. I decided to do a 10-fold
cross-validation to find the best model. After that I used a data augmentation technique to increase
the number of examples. As shown in the notebooks, the results improved a lot.

Three models were implemented: 
- SVM
- CNN
- MLP

All the implementation details are documented in the notebooks.

Scaling up
-----------

In order to scale this solution to thousands of requests a day one idea is to
use a cache mechanism, like redis, varnish or memcached. Another idea is to
'dockerize' the app and use kubernets to scale it horizontally. The goal is to
have a load balanced such as Ngnix to redirect the requests. 


TODOs
------------
- [ ] Deploy to Heroku
- [ ] Dockerize the app
- [ ] Add memcached or Varnish to cache api calls
- [ ] Add a new endpoint to retrain a model
- [ ] Make the model name optional on the predict 
endpoint so that the result of all models is returned
