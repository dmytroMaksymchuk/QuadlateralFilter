import numpy as np
import plotly.graph_objects as go

from QuadlateralFilter.quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from QuadlateralFilter.trilateral.trilateralFilter2D import trilateral_filter_2d

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

    fig = go.Figure(data=[go.Surface(z=filtered_Z - Z)])

    # Add a scatter plot for hover effect
    fig.add_trace(go.Surface(z= Z + 100))

    # Update layout
    fig.update_layout(title='qqqqqq', autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90),
                      scene=dict(
                          hovermode='closest',
                      ))
    fig.update_layout(clickmode='event+select')

    def update_hover():
        return go.Figure(data=[go.Surface(z=filtered_Z - Z)])

    fig.data[0].on_click(update_hover())

    # Show plot
    fig.show()