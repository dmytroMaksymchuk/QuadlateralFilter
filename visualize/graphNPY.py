import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    im1 = np.load('../HDR/images/resultImages/memorial/more/npy/qu_detail_layer.npy')
    im2 = np.load('../HDR/images/resultImages/memorial/more/npy/t_detail_layer.npy')

    diff = im1 - im2

    plt.figure(figsize=(12, 6))
    plt.imshow(diff, cmap='bwr', vmin=-0.2, vmax=0.2, interpolation='nearest')
    plt.colorbar()
    plt.title('Difference with Interpolation')
    plt.show()
