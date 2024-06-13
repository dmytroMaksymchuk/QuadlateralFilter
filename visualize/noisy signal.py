# create a noisy signal

import numpy as np
import matplotlib.pyplot as plt

# if __name__ == '__main__':
#     t = np.linspace(0, 1, 500)
#     y_const_low = np.ones(50) * 0.2
#
#     #log graph for next 500 points
#     y_log = np.logspace(0.2, 1.2, 250)
#     y_const_high = np.ones(200) * 7
#
#     y = np.concatenate((y_const_low, y_log, y_const_high))
#
#     # add noise
#     noise = np.random.normal(0, 0.3, y.shape)
#     y += noise
#
#     plt.plot(t, y)
#     plt.show()

if __name__ == '__main__':
    #draw gauss normal curve with higher sigma
    x = np.linspace(-5, 5, 1000)
    y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    y2 = np.exp(-x**2 / 8) / np.sqrt(8 * np.pi)
    #splt.plot(x, y)
    plt.plot(x, y2)
    plt.show()