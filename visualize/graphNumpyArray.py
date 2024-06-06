import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    #load numpy array
    arr = np.load('../visualize/resultsStatue.npy')
    sigmaSpatials = [4, 6, 8, 10]
    sigmaIntensities = [5, 15, 20, 30, 40]
    bilateral_noise = np.array(
        [[arr[i][j][0] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    trilateral_noise = np.array(
        [[arr[i][j][1] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    quadlateral_noise = np.array(
        [[arr[i][j][2] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])

    # Visualization with subplots for each sigma spatial for noise_estimation

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    for i, sigmaSpatial in enumerate(sigmaSpatials):
        row = i // 2
        col = i % 2
        ax = axes[row][col]
        ax.plot(sigmaIntensities, bilateral_noise[i], label='Bilateral')
        ax.plot(sigmaIntensities, trilateral_noise[i], label='Trilateral')
        ax.plot(sigmaIntensities, quadlateral_noise[i], label='Quadlateral')
        ax.set_title('Noise Estimation for Sigma Spatial: ' + str(sigmaSpatial))
        ax.set_xlabel('Sigma Intensity')
        ax.set_ylabel('Noise Estimation')
        ax.legend()
        ax.grid(True)
    plt.show()

