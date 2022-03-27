""" Main FastAPI app """
import logging
import os
from pydantic import BaseModel, Field

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bayesian import run_one_1d_bayesian_optimization

logger = logging.getLogger(__name__)

version = "0.0.1"
description = """
Simple API that hosts code for running bayesian optimization.
Currently only 1D bayesian optimization has been implemented but the plan is to expand to 2D.
"""
app = FastAPI(
    title="Materiom Active Learning API",
    version=version,
    description=description,
    contact={
        "name": "Materiom",
        "github": "https://github.com/materiom/ActiveLearningAPI"
    },
)


# CORS - https://fastapi.tiangolo.com/tutorial/cors/
# CORS or "Cross-Origin Resource Sharing" refers to the situations
# when a frontend running in a browser has JavaScript code that
# communicates with a backend, and the backend is in a different
# "origin" than the frontend.
origins = os.getenv("ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Point2D(BaseModel):
    x: float = Field(..., description="x value")
    y: float = Field(..., description="x value")


class OneErrorBar(BaseModel):
    mu: float = Field(..., description="Mean, the expect value")
    sigma: float = Field(..., description="standarad deviation, the std on the expected value. "
                                          "I.e how confident of the value of 'mu' are we.")
    x: float = Field(..., description="x value")


class ResponseOneDimension(BaseModel):
    error_bars: List[OneErrorBar] = Field(..., description="List of error bars ready for plotting")
    suggestion_x: float = Field(..., description="The best suggestion for x. "
                                                 "'best' is defined as the best guess for 'x' that maximises the "
                                                 "variable 'y'")


class InputOneDimension(BaseModel):
    data: List[Point2D] = Field(..., description="List of data")


@app.get("/")
async def get_api_information():
    """Get information about the API itself"""

    logger.info("Route / has be called")

    return {
        "title": "Materiom API",
        "version": version,
        "description": description,
        "documentation": "/docs",
    }


@app.get("/run_one_1d_bayesian_optimization", response_model=ResponseOneDimension)
async def one_1d_bayesian_optimization(inputs: InputOneDimension):
    """
    get one 1d bayesian optimization results

    :param inputs: observed data
    :return: distribution and next suggestion
    """
    data = [d.dict() for d in inputs.data]

    mu, sigma, grid, suggestion_x = run_one_1d_bayesian_optimization(data=data)

    error_bars = []
    for i in range(len(mu)):
        error_bars.append(OneErrorBar(mu=mu[i], sigma=sigma[i], x=grid[i]))

    response_one_dimension = ResponseOneDimension(error_bars=error_bars, suggestion_x=suggestion_x)

    return response_one_dimension
