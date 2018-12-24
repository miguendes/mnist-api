MNIST Api
==========================

[![Build Status](https://travis-ci.org/mendesmiguel/mnist-api.svg?branch=master)](https://travis-ci.org/mendesmiguel/mnist-api)
[![codecov](https://codecov.io/gh/mendesmiguel/mnist-api/branch/master/graph/badge.svg)](https://codecov.io/gh/mendesmiguel/mnist-api)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.7/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


A Flask REST API for handwritten digit recognition using machine learning models.

The goal of this project is to develop a REST Api that receives an image and returns a
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
- opencv-python
- Pillow
- gunicorn
- seaborn

To install Pipenv and Python 3.6 without messing up your environment, in case you don't have them installed,
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

Pipenv will spin up a virtualenv and install the dependencies based on a `Pipenv` file inside the root of
the project. 

``` {.sourceCode .bash}
$ cd mnist-api/
$ pipenv install
```

#### (Optional) Docker

You can use the `Dockerfile` provided to run an already setup environment.

``` {.sourceCode .bash}
docker build -t docker-mnist-api:latest . 
docker run -d -p 5000:5000 docker-mnist-api
```


How to run the app locally
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
$ curl -d '{"model":"cnn", "image": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAB/UlEQVR4nO3Uv+txURwH8ONHRLosUuRHSSFFVovh+lUYLBaL/0JKSTYMdiVhubk2XWUyMhik7sDiKgpJhKv08Qy3vo+er+f58qWe5fsez6fP63zOOXUQ+sn/islkKhaL9Xq92+0uFot2u329XkmSDAQC3+FcLtdkMjmdTnAvLMuGQqE/WoT/FiORCEEQfD6fpunhcLharRiG4UperxfHcZFIpNPpnkPj8Xij0SAIotVqnc/n25JarcZxHACOx+PX572NXC4XCu9s7Ha7l8slAPR6vefEuxEIBIlEYjqdAsBsNjObza+KBoOBIAjuiRiGUSgUL3FGo9Hv9/f7fQBYrVaRSOTutTwaDMOSyeThcOAGpGna6XS+NCBCyOv17vd7ThyPxxaL5VURIaTX60ejEcuynLvdbmOx2BtchJDD4chms7vdDgA2m41KpXqPixBKp9MAcLlcwuHwe0SZTDafzwGg0+k82uPz+fL5vFQqvVuVSCTNZpO71s8/yF9D0zTDMHa7/XNJKpUWCgVOrFQqPB7vUbRarQJAqVQSi8W36zabjaIoThwMBlar9Uvq954ej4eiKD6fn8vl1us1txiNRrVarVKpRAiVy+VMJjOdTh8dEyGk0WhIkrz7E+92u1QqhWHYE9xHLBZLrVbbbrcfXD6fDwaDEonkO9xP3phfpiAnDIM7k5QAAAAASUVORK5CYII=
"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predict/
```

#### Warning: this api works only with RGB PNG images. Up to now no other image format is supported.

### Heroku

The app has also been deployed to Heroku and it's available on:

https://miguel-mnist-api.herokuapp.com/

Example of prediction:
``` {.sourceCode .bash}
$ curl -d '{"model":"cnn", "image": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAB/UlEQVR4nO3Uv+txURwH8ONHRLosUuRHSSFFVovh+lUYLBaL/0JKSTYMdiVhubk2XWUyMhik7sDiKgpJhKv08Qy3vo+er+f58qWe5fsez6fP63zOOXUQ+sn/islkKhaL9Xq92+0uFot2u329XkmSDAQC3+FcLtdkMjmdTnAvLMuGQqE/WoT/FiORCEEQfD6fpunhcLharRiG4UperxfHcZFIpNPpnkPj8Xij0SAIotVqnc/n25JarcZxHACOx+PX572NXC4XCu9s7Ha7l8slAPR6vefEuxEIBIlEYjqdAsBsNjObza+KBoOBIAjuiRiGUSgUL3FGo9Hv9/f7fQBYrVaRSOTutTwaDMOSyeThcOAGpGna6XS+NCBCyOv17vd7ThyPxxaL5VURIaTX60ejEcuynLvdbmOx2BtchJDD4chms7vdDgA2m41KpXqPixBKp9MAcLlcwuHwe0SZTDafzwGg0+k82uPz+fL5vFQqvVuVSCTNZpO71s8/yF9D0zTDMHa7/XNJKpUWCgVOrFQqPB7vUbRarQJAqVQSi8W36zabjaIoThwMBlar9Uvq954ej4eiKD6fn8vl1us1txiNRrVarVKpRAiVy+VMJjOdTh8dEyGk0WhIkrz7E+92u1QqhWHYE9xHLBZLrVbbbrcfXD6fDwaDEonkO9xP3phfpiAnDIM7k5QAAAAASUVORK5CYII=
"}' -H "Content-Type: application/json" -X POST https://miguel-mnist-api.herokuapp.com/predict/
```
Expected response:
``` {.sourceCode .bash}
{"prediction":3}
```


Methodology
------------

The task has an important restriction: only 500 images to train a model on, including the test set.
Since this amount is so little to train a good classifier, I decided to do a 10-fold
cross-validation to find the best model. After that, I used a data augmentation technique to increase
the number of examples and the results improved a lot, as shown in the notebooks.

Three models were implemented: 
- SVM
- CNN
- MLP

All the implementation details are documented in the notebooks.

Once we collect more data, we can test the performance of the current models and retrain them from time to time.

### How to run the notebooks

To run the jupyter notebooks where I performed the EDA and trained the models, do:

``` {.sourceCode .bash}
$ cd mnist-api/
$ pipenv shell
$ jupyter notebook
```

All notebooks are available on the `notebooks` directory.

Scaling up
-----------

### Caching with Redis

I performed a series of benchmarks using redis as caching mechanism and Apache Bench v2.3 to run the benchmark.
The code is available on branch `feature/caching`.

The difference between a cached API and a non-cached API is impressive. 
I simulated a scenario where 100 clients are making 100 requests each.
The results are shown bellow:

#### No cache
```
$ ab -p post_content.txt -T application/json -c 100 -n 100 http://localhost:8000/predict/
....
Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        2    3   0.6      3       4
Processing:   714 38391 22153.1  38658   76232
Waiting:      714 38390 22153.1  38657   76232
Total:        718 38393 22152.5  38660   76234
....

```

#### Caching with Redis
The caching has a timeout of 50 seconds.
```
$ ab -p post_content.txt -T application/json -c 100 -n 100 http://localhost:8000/predict/

....
Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        1    2   0.6      2       4
Processing:   706  758  27.2    760     803
Waiting:      705  758  27.2    760     803
Total:        709  761  26.5    762     804
....

```

### Aditional approaches

Another idea is to use kubernets to scale it horizontally. In this setting a good idea is to
have a load balancer such as Ngnix to handle the requests.


TODOs
------------
- [x] Make the weights path on config file plataform agnostic
- [x] Add tests coverage report 
- [x] Add memcached or Varnish to cache api calls
- [ ] Add a new endpoint to retrain a model
- [x] Benchmark the API using Apache Bench (ab) and test with a cluster of containers and a load balancer
- [ ] Make the model name optional on the predict endpoint so that the result of all models is returned
