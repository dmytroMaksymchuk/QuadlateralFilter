import numpy as np
import plotly.graph_objects as go

from QuadlateralFilter.quadlateral.quadlateralFilter2D import quadrateral_filter_2d

if __name__ == '__main__':
    # Define grid of x and y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Define parameters of the parabolic surface equation: z = ax^2 + by^2 + c
    a = 2
    b = -2
    c = 0

    # Calculate z values using the parabolic equation
    Z = a * X ** 2 + b * Y ** 2 + c

    # noised_Z = add_gauss_noise_shape(Z, sigma=1)
    noised_Z = Z
    filtered_Z = quadrateral_filter_2d(noised_Z, 3)

    fig = go.Figure(data=[go.Surface(z = filtered_Z - Z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    fig.update_layout(title='Quadlateral filter', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))


    def click_callback(trace, points, selector):
        if points.point_inds:
            # Get the clicked point coordinates
            x = points.xs[0]
            y = points.ys[0]
            z = filtered_Z[int(points.point_inds[0][0]), int(points.point_inds[0][1])]

            # Add a vertical line at the clicked point
            fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, z],
                                       mode='lines',
                                       line=dict(color='red', width=3),
                                       name='Vertical Line'))


    # Register the callback function for click event
    fig.data[0].on_click(click_callback)

    fig.show()