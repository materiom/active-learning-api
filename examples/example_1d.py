from src.bayesian import run_one_1d_bayesian_optimization

# fake data
data = [{"x": 1, "y": 0},
        {"x": 1.5, "y": 1.5},
        {"x": 2, "y": 1},
        {"x": 3, "y": -1}]

mu, sigma, grid, s = run_one_1d_bayesian_optimization(data=data, limits={"x": (0, 4)})

# plot
import plotly.graph_objects as go

traces = []
traces.append(go.Scatter(x=grid, y=mu, mode='lines', name='mu'))
traces.append(go.Scatter(
    x=[s,s],
    y=[-2,3],
mode='lines',name="sugesstion"))
traces.append(go.Scatter(x=grid, y=mu-sigma,
    fill=None,
    mode='lines',
    line_color='indigo', name='mu+-sigma'
    ))
traces.append(go.Scatter(
    x=grid,
    y=mu+sigma,
    fill='tonexty', # fill area between trace0 and trace1
    mode='lines', line_color='indigo',showlegend=False))
fig = go.Figure(data=traces)

fig.update_layout(
    title="1D Bayesian Optimization",
    xaxis_title="Optimization space",
    yaxis_title="Maximum",
)

fig.show()
