import numpy as np
import matplotlib.pyplot as plt
import GPy
from myGP import MyGP

def create_stepFunction():
    """Create Step functions."""
    # Create Data
    num_of_xPoints = 10
    x_train = np.linspace(-3, 3, num_of_xPoints)
    y_train = []
    sig_noise = np.random.normal(0, 0.04, size=5)

    for sig in sig_noise:
        sim = (1 + sig) * np.heaviside(x_train, 1)
        y_train.append(sim)
        
    x_train = np.meshgrid(x_train, sig_noise)[0].T
    y_train = np.array(y_train).T

    return x_train, y_train

def gpy_for_step(x_train, y_train, x_test):
    """Create GPy for step function.

    Args:
        x_train (np.ndarray): x values of training data
        y_train (np.ndarray): y values of training data
        x_test (np.ndarray): x values for prediction

    Return:
        (mu, cov) mean an covariance of GP

    """
    y_train = y_train.flatten().reshape(-1, 1)
    x_train = x_train.flatten().reshape(-1, 1)

    kernel = GPy.kern.RBF(input_dim=1, variance=1.)
    m = GPy.models.GPRegression(x_train, y_train, kernel)

    m.optimize(messages=True)
    m.optimize_restarts(num_restarts=10)

    # get mean and cov
    mu, cov = m.predict(x_test, full_cov=True, include_likelihood=False)

    return m, mu, cov


def gp_for_step(x_train, y_train, x_test, axes=None, label='sim'):
    """Obtain predictive posterior of my GP on training data.

    Args:
        x_train (np.ndarray): x values of training data
        y_train (np.ndarray): y values of training data
        x_test (np.ndarray): x values for prediction
        axes (plt): matplotlib axes
        label (str): legend label for plot

    Return:
        (mu, cov) mean an covariance of GP

    """
    y_train = y_train.flatten().reshape(-1, 1)
    x_train = x_train.flatten().reshape(-1, 1)

    # Initialize GP
    sigma = 0.2
    length = 1.3
    sigma_noise = 0.04
    GP = MyGP(x_train, y_train, [sigma, length, sigma_noise])

    # Optimize hyperparameter
    GP.optimize_logLikelihood()

    # Get the posterior for fixed hyperparameters
    (mu, cov) = GP.predictive_posterior(x_test)
    GP.sample_from_posterior(x_test, axes=axes, label=label)

    return mu, cov


if __name__ == "__main__":
    # Create some data
    x_train, y_train = create_stepFunction()
    
    # Create test data
    N = 200
    x_test = np.linspace(np.min(x_train[:, 0]),
                         np.max(x_train[:, 0]), N).reshape(-1, 1)

    # Create GP with kernel
    # =====================
    # using gpy library
    gp_model, mu, cov = gpy_for_step(x_train, y_train, x_test)
    
    _ = gp_model.plot() # plot posterior

    # sample from multivariate normal distribution
    y_test = gp_model.posterior_samples_f(x_test, full_cov=True, size=10)
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'kx', label='data')
    for i in range(10):
        ax.plot(x_test[:], y_test[:,:,i], '-')

    # using my own GP
    # ===============
    _, ax2 = plt.subplots()
    mu, cov = gp_for_step(x_train, y_train, x_test, axes=ax2, label='GP')

    plt.show()