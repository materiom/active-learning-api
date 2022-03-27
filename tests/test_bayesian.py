from bayesian import run_one_1d_bayesian_optimization


def test_run_one_1d_bayesian_optimization():
    data = [{"x": 1, "y": 0}, {"x": 2, "y": 1}, {"x": 3, "y": -1}]

    mu, sigma, grid, s = run_one_1d_bayesian_optimization(data=data, limits={"x": (2, 4)})

    assert len(mu) == 1000
    assert len(sigma) == 1000

    assert s >= 2
    assert s <= 4
