""" Test for main app """
from fastapi.testclient import TestClient

from main import app, version, OneData, InputOneDimension

client = TestClient(app)


def test_read_main():
    """Check main route works"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == version


def test_read_latest_one_gsp():
    """Check main GB/pv/gsp/{gsp_id} route works"""

    data = [{"x": 1, "y": 0}, {"x": 2, "y": 1}, {"x": 3, "y": -1}]

    input = InputOneDimension(data=[OneData(**d) for d in data])

    response = client.get("/run_one_1d_bayesian_optimization",json=input.dict())
    print(response.text)
    assert response.status_code == 200



