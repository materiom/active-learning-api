""" Main FastAPI app """
import logging
from pydantic import BaseModel, Field

from typing import List

from fastapi import FastAPI
from bayesian import run_one_1d_bayesian_optimization

logger = logging.getLogger(__name__)

version = "0.0.1"
description = """
"""
app = FastAPI(
    title="Materiom Active Learning API",
    version=version,
    description=description,
    contact={
        "name": "Materiom",
    },
)


class OneData(BaseModel):
    x: float = Field(..., description="x value")
    y: float = Field(..., description="x value")


class OneErrorBar(BaseModel):
    mu: float = Field(..., description="Mean")
    sigma: float = Field(..., description="standarad deviation")
    x: float = Field(..., description="x value")


class ResponseOneDimension(BaseModel):
    error_bars: List[OneErrorBar] = Field(..., description="List of error bars ready for plotting")
    suggestion_x: float = Field(..., description="The best suggestion for x")


class InputOneDimension(BaseModel):
    data: List[OneData] = Field(..., description="List of data")


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

    data = [d.dict() for d in inputs.data]

    mu, sigma, grid, suggestion_x = run_one_1d_bayesian_optimization(data=data)

    error_bars = []
    for i in range(len(mu)):
        error_bars.append(OneErrorBar(mu=mu[i], sigma=sigma[i], x=grid[i]))

    response_one_dimension = ResponseOneDimension(error_bars=error_bars, suggestion_x=suggestion_x)

    return response_one_dimension
