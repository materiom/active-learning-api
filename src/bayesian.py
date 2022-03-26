from bayes_opt import BayesianOptimization
import numpy as np
from typing import List, Optional


N_grid_points = 1000

# This jupyter notebook helped me
# https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb


def run_one_1d_bayesian_optimization(data: List[dict], limits: Optional[dict] = None):
    """
    Run one dimension bayesian optimization

    :param data: observed data
    :param limits: optional limits for the search space.
        This should be a dictionary with a key of
        'x' and then a minimum and maximum value for x. For example {'x': [1,2]}.
        We might want to do this as we know the solution lies in a certain space
    :return: 'mu, sigma, grid, suggestion_x'
        - mu the expected return given an x.
            This is an array which each point linked to 'grid'
        - sigma the standard deviation on the estimate of 'mu'.
            This is an array which each point linked to 'grid'
        - grid is an array for which 'mu' and 'sigma' lie on.
        - suggestion_x is the next suggestion for x.
            This is the highest value of mu+sigma.
    """
    # observed data
    x_obs = np.array([[d["x"]] for d in data])
    y_obs = np.array([d["y"] for d in data])

    # create fake function black_box_function
    # this function is not needed, but the package requires this.
    # This is because the package is designed to find maximums of solve analytical equations
    black_box_function = lambda x: x

    if limits is None:
        limits = {"x": (np.min(x_obs), np.max(x_obs))}

    # set upt bayesian optimization
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=limits,
        random_state=1,  # hard-coded for the moment
    )

    # set grid we want to return values
    grid = np.linspace(limits["x"][0], limits["x"][-1], N_grid_points).reshape(-1, 1)

    # fit the observed points
    optimizer._gp.fit(x_obs, y_obs)

    # make predictions
    mu, sigma = optimizer._gp.predict(grid, return_std=True)

    # make suggestion
    suggestion_idx = np.argmax(mu + sigma)
    suggestion_x = grid[suggestion_idx,0]

    return mu, sigma, grid[:,0], suggestion_x
