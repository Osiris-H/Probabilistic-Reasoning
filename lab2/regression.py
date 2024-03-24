import numpy as np
import matplotlib.pyplot as plt
import util
from scipy.stats import multivariate_normal


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """

    mean = np.array([0, 0])
    cov = np.array([[beta, 0], [0, beta]])

    a0_range = np.linspace(-1, 1, 100)
    a1_range = np.linspace(-1, 1, 100)
    a0, a1 = np.meshgrid(a0_range, a1_range)
    pos = np.dstack((a0, a1)).reshape(-1, 2)

    density_values = util.density_Gaussian(mean, cov, pos).reshape(a0.shape)

    plt.figure(figsize=(6, 6))
    plt.contour(a0, a1, density_values, levels=10, cmap='viridis')
    plt.colorbar()
    plt.title(f'Prior Distribution p(a) contours with β = {beta}')
    plt.plot(-0.1, -0.5, 'ro')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.grid(True)
    plt.axis('equal')
    # plt.show()
    plt.savefig('prior.pdf', format='pdf')


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    sigma_w_inv = np.diag([1 / sigma2] * x.shape[0])
    sigma_a_inv = np.array([[1 / beta, 0], [0, 1 / beta]])

    ones_column = np.ones((x.shape[0], 1))
    X = np.hstack((ones_column, x))

    mu = np.linalg.inv(sigma_a_inv + X.T @ sigma_w_inv @ X) @ (X.T @ sigma_w_inv @ z)
    Cov = np.linalg.inv(sigma_a_inv + X.T @ sigma_w_inv @ X)

    return (mu, Cov)


def plotPosteriorDistribution(mu, Cov):

    mu = mu.reshape(-1)

    a0_range = np.linspace(-1, 1, 100)
    a1_range = np.linspace(-1, 1, 100)
    a0, a1 = np.meshgrid(a0_range, a1_range)
    pos = np.dstack((a0, a1)).reshape(-1, 2)

    density_values = util.density_Gaussian(mu, Cov, pos).reshape(a0.shape)

    plt.figure(figsize=(6, 6))
    plt.contour(a0, a1, density_values, levels=5, cmap='viridis')
    plt.colorbar()
    plt.plot(-0.1, -0.5, 'ro')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('Posterior Distribution p(a|x,z)')
    plt.grid(True)
    plt.xlabel('a0')
    plt.ylabel('a1')
    # plt.show()
    plt.savefig('posterior5.pdf', format='pdf')


def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """

    X = np.vstack((np.ones(len(x)), x))
    # print(X)
    mu_test = mu.T @ X
    cov_test = sigma2 + X.T @ Cov @ X
    std_test = np.sqrt(np.diag(cov_test))
    # print(mu_test.shape)
    # print(cov_test)

    plt.figure(figsize=(10, 8))

    plt.errorbar(x, mu_test.reshape(-1), yerr=std_test, fmt='o', label='Predictions with 1 Std Dev')
    plt.scatter(x_train, z_train, color='red', label='Training Samples')

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('Input')
    plt.ylabel('Target')
    plt.title('Predictions and Training Samples')
    plt.legend()
    # plt.show()
    plt.savefig('predict5.pdf', format='pdf')


if __name__ == '__main__':
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]

    # known parameters 
    sigma2 = 0.1
    beta = 1

    # number of training samples used to compute posterior
    ns = 100

    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]

    # prior distribution p(a)
    # priorDistribution(beta)

    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x, z, beta, sigma2)
    plotPosteriorDistribution(mu, Cov)

    # distribution of the prediction
    predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)
