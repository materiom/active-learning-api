# Active Learning API

Simple API that hosts code for running bayesian optimization.
Currently only 1D bayesian optimization has been implemented but the plan is to expand to 2D.

# Documentation

We use [FastAPI](https://fastapi.tiangolo.com/).
Documentation can be viewed at `/docs`. This is automatically generated from the code.

There is current one endpoint `run_one_1d_bayesian_optimization` that runs a bayesian optimization in one dimension 
and returns distribution and the next suggestion.

# Examples

Please see `examples/example_1d.py` which a very small example using the code and plotting it
![example 1](./example/example_1d.png)

# Setup and Run

This can be done it two different ways: With Python or with Docker.

## Python

### Create a virtual env

```bash
python3 -m venv ./venv
source venv/bin/activate
```

### Install Requirements and Run

```bash
pip install -r requirements.txt
cd src && uvicorn main:app --reload
```

### Local pytest

To run local pytests you need to
1. add `src` to python path `export PYTHONPATH=$PYTHONPATH:./src`
2. run pytests: `pytest`

## Docker

1. Make sure docker is installed on your system.
2. Use `docker-compose up`
   in the main directory to start up the application.
3. You will now be able to access it on `http://localhost:80`
