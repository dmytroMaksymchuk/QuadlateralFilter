# Import packages
import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input, dash, ctx
import plotly.graph_objects as go
import cv2 as cv

from helpers.gaussianHelper import add_gauss_noise_2d_image
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d

def get_data():
    x = np.linspace(-2 , 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.cos(X) * np.cos(Y) * 250
    Z = Z.clip(0, 180).astype(np.uint8)

    sigmaSpatial = 3
    sigmaIntensity = 20

    # Z = Z.clip(0, 255).astype(np.uint8)
    # Z = np.where(Z > 70, 200, 10)

    # Z = cv.imread('../images/golf_snippet_low_contrast.png', cv.IMREAD_GRAYSCALE)

    noised_Z = add_gauss_noise_2d_image(Z, 3)
    # noised_Z = Z

    noised_Z = noised_Z.clip(0, 255).astype(np.uint8)

    bilateral = cv.bilateralFilter(noised_Z, 31, sigmaIntensity, sigmaSpatial)
    result, quad, uncert = quadrateral_filter_2d(noised_Z, sigmaSpatial, sigmaIntensity)
    inclusion005 = quadrateral_filter_2d(noised_Z, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.05)[0].clip(0, 255).astype(np.uint8)
    inclusion01 = quadrateral_filter_2d(noised_Z, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.1)[0].clip(0, 255).astype(np.uint8)
    inclusion03 = quadrateral_filter_2d(noised_Z, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.2)[0].clip(0, 255).astype(np.uint8)
    result = result.clip(0, 255).astype(np.uint8)

    trilat = trilateral_filter_2d(noised_Z, sigmaSpatial, sigmaIntensity).clip(0, 255).astype(np.uint8)

    return Z, noised_Z, bilateral, trilat, result, quad, inclusion005, inclusion01, inclusion03


def get_fig(z_data, diffGraph=False, uncert=False):

    if diffGraph:
        fig = go.Figure(data=[go.Surface(z=z_data, cmin=0, cmax=10)])
    elif uncert:
        fig = go.Figure(data=[go.Surface(z=z_data, cmin=0, cmax=1)])
    else:
        fig = go.Figure(data=[go.Surface(z=z_data, cmin=0, cmax=255)])

    fig.update_layout(
        title='Mt Bruno Elevation',
        autosize=True,
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        clickmode='event+select'  # Enable click events on the graph
    )
    fig.update_layout(scene=dict(zaxis=dict(range=[-10, 255])))
    fig.update_layout(scene_aspectmode='cube')
    return fig

def get_option_z_data(curr_graph_option):
    if curr_graph_option == 'noise':
        z_data = noised
    elif curr_graph_option == 'filtered':
        z_data = filtered
    elif curr_graph_option == 'diff':
        z_data = np.abs(filtered.astype(np.float32) - original)
    elif curr_graph_option == 'diff_trilat':
        z_data = np.abs(trilat.astype(np.float32) - original)
    elif curr_graph_option == 'diff_bilateral':
        z_data = np.abs(bilateral.astype(np.float32) - original)
    elif curr_graph_option == 'trilateral':
        z_data = trilat
    elif curr_graph_option == 'bilateral':
        z_data = bilateral
    elif curr_graph_option == 'incl05':
        z_data = inclusion05
    elif curr_graph_option == 'incl2':
        z_data = inclusion2
    elif curr_graph_option == 'incl5':
        z_data = inclusion5
    elif curr_graph_option == 'diff_incl05':
        z_data = np.abs(inclusion05.astype(np.float32) - original)
    elif curr_graph_option == 'diff_incl2':
        z_data = np.abs(inclusion2.astype(np.float32) - original)
    elif curr_graph_option == 'diff_incl5':
        z_data = np.abs(inclusion5.astype(np.float32) - original)
    else:
        z_data = original
    return z_data

def update_figure_coord(x_coord, y_coord, z_data):

    if x_coord is None or y_coord is None:
        return get_fig(z_data)

    x = float(x_coord)
    y = float(y_coord)
    z = quad[int(y), int(x)]  # Getting the corresponding z value from the dataset

    kernel_size = 3 * 3
    xLB = max(int(x) - kernel_size, 0)
    yLB = max(int(y) - kernel_size, 0)
    xUB = min(int(x) + kernel_size, quad.shape[1])
    yUB = min(int(y) + kernel_size, quad.shape[0])

    plane_x = np.linspace(xLB, xUB, xUB - xLB + 1)
    plane_y = np.linspace(yLB, yUB, yUB - yLB + 1)
    plane_x, plane_y = np.meshgrid(plane_x, plane_y)

    midZ_x = kernel_size + min(int(x) - kernel_size, 0)

    midZ_y = kernel_size + min(int(y) - kernel_size, 0)

    scatter_point = go.Scatter3d(x=[x_coord], y=[y_coord], z=[z[midZ_y][midZ_x]])

    fig = go.Figure(data=[
        go.Surface(z=z_data, opacity= 0.8),
        go.Surface(x=plane_x, y=plane_y, z=z, opacity=1, colorscale='Viridis'),
        scatter_point
    ])

    fig.update_layout(
        title='Mt Bruno Elevation with Plane',
        autosize=True,
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        clickmode='event+select'  # Enable click events on the graph
    )
    fig.update_layout(scene=dict(zaxis=dict(range=[-10, 255])))
    fig.update_layout(scene_aspectmode='cube')
    return fig


original, noised, bilateral, trilat, filtered, quad, inclusion05, inclusion2, inclusion5 = get_data()
app = Dash()
app.layout = html.Div([
    html.Div([
        dcc.Input(id='x-coord', type='number', placeholder='Enter X Coordinate'),
        dcc.Input(id='y-coord', type='number', placeholder='Enter Y Coordinate'),
        dcc.RadioItems(options=['original', 'noise', 'bilateral', 'diff_bilateral',
                                'trilateral', 'diff_trilat', 'filtered', 'diff',
                                'incl05', 'incl2', 'incl5', 'diff_incl05', 'diff_incl2', 'diff_incl5'],
                       value='original', id='choose-graph'),
        dcc.Graph(id='elevation-graph', figure=get_fig(original))
    ])
])


@app.callback(
    Output('elevation-graph', 'figure'),
    [
        Input('choose-graph', 'value'),
        Input('x-coord', 'value'),
        Input('y-coord', 'value')
    ]
)
def update_graph(graph_chosen, x_coord, y_coord):
    print("Updating graph", ctx.triggered_id)
    changed_input = ctx.triggered_id
    z_data = get_option_z_data(graph_chosen)
    if changed_input == 'x-coord' or changed_input == 'y-coord':
        return update_figure_coord(x_coord, y_coord, z_data)
    else:
        return get_fig(z_data, graph_chosen.startswith('diff'), graph_chosen == 'uncert')


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port= 8051)
