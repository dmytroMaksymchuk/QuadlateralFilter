# Import packages
import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from QuadlateralFilter.quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from QuadlateralFilter.trilateral.trilateralFilter2D import match_shape


def get_data():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.sin(X) * np.cos(Y) * 10

    # noised_Z = add_gauss_noise_shape(Z, sigma=1)
    noised_Z = Z
    result, quad = quadrateral_filter_2d(noised_Z, 3)
    return Z, quad


def get_fig(z_data):
    fig = go.Figure(data=[go.Surface(z=z_data)])

    fig.update_layout(
        title='Mt Bruno Elevation',
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        clickmode='event+select'  # Enable click events on the graph
    )
    return fig


def update_figure(x_coord, y_coord):
    if x_coord is None or y_coord is None:
        return get_fig(z_data)

    x = float(x_coord)
    y = float(y_coord)
    z = quad[int(y), int(x)]  # Getting the corresponding z value from the dataset

    plane_x = np.linspace(x - 9, x + 10, 19)
    plane_y = np.linspace(y - 9, y + 10, 19)
    plane_x, plane_y = np.meshgrid(plane_x, plane_y)

    print("Z is:", z)

    scatter_point = go.Scatter3d(x=[x_coord], y=[y_coord], z=[z[9][9]])

    fig = go.Figure(data=[
        go.Surface(z=z_data),
        go.Surface(x=plane_x, y=plane_y, z=z, opacity=0.5, colorscale='Viridis'),
        scatter_point
    ])

    fig.update_layout(
        title='Mt Bruno Elevation with Plane',
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        clickmode='event+select'  # Enable click events on the graph
    )
    return fig


app = Dash()
z_data, quad = get_data()
app.layout = html.Div([
    html.Div([
        dcc.Input(id='x-coord', type='number', placeholder='Enter X Coordinate'),
        dcc.Input(id='y-coord', type='number', placeholder='Enter Y Coordinate'),
        dcc.Graph(id='elevation-graph', figure=get_fig(z_data))
    ])
])


@app.callback(
    Output('elevation-graph', 'figure'),
    [Input('x-coord', 'value'),
     Input('y-coord', 'value')]
)
def update_graph(x_coord, y_coord):
    return update_figure(x_coord, y_coord)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
