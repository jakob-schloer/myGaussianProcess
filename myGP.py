"""This is an implementation of an Gaussian Process with an RBF kernel."""
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import scipy.optimize as opt


class MyGP():
    """This is a self coded Gaussian Process with an RBF kernel.

    Be cautios with the results, it might be really buggy.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray,
                 hyperparams: list):
        """Initialize my GP.

        Args:
            x_train (np.ndarray): x values of the training data of the GP
            y_train (np.ndarray): y values of the training data
            hyperparams (list): initial hyperparameter of the kernel,
                i.e. [sigma, length, noise]

        """
        self.x_train = x_train
        self.y_train = y_train
        # some variables
        self.m = np.shape(x_train)[0]
        self.sigma = hyperparams[0]
        self.length = hyperparams[1]
        self.noise = hyperparams[2]
        self.eps = 1.0e-7   # to enable positive definite matrices

        # Get pairwise euclidian distance matrix
        self.X = self.square_distanceMatrix(x_train, x_train)

    def square_distanceMatrix(self, x1: np.ndarray, x2: np.ndarray):
        """Obtain the Pairwise Euclidian Distance Matrix.

        Args:
            x1 (np.ndarray): one dimensional input
            x2 (np.ndarray): one dimensional input

        Return:
            np.ndarray: Pairwise Euclidian Distance Matrix

        """
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
        return sqdist

    def rbf_kernel(self, sqdist: np.ndarray, params: list):
        """Calculate RBF kernel matrix.

        Args:
            sqdist (np.ndarray): Euclidian Distance Matrix sqdist(x1, x2)
            params (list): Includes hyperparameter of the kernel [sigma, length]

        Return:
            np.ndarray: RBF Kernel matrix

        """
        sigma = params[0]
        length = params[1]
        K = sigma**2 * np.exp(-1/(length) * sqdist)
        return K

    def predictive_posterior(self, x_test: np.ndarray):
        """Calculate the predictive posterior for the test data.

        Args:
            x_test (np.ndarray): x values where the GP should give a prediction

        Return:
            (mu, cov): the mean and covariance of the posterior gaussian

        """
        # some variables
        m = self.m
        n = np.shape(x_test)[0]
        sigma = self.sigma
        length = self.length
        noise = self.noise

        # Get pairwise euclidian distance matrix
        X = self.X
        X_s = self.square_distanceMatrix(self.x_train, x_test)
        X_ss = self.square_distanceMatrix(x_test, x_test)

        # Create covariance matrix of joint probability
        K = self.rbf_kernel(X, [sigma, length]) + np.diag(noise**2*np.ones(m))
        K_s = self.rbf_kernel(X_s, [sigma, length])
        K_ss = self.rbf_kernel(X_ss, [sigma, length])

        # Compute the mean of the predictive posterior
        L = np.linalg.cholesky(K + self.eps*np.eye(m))
        Lk = np.linalg.solve(L, K_s)
        Lf = np.linalg.solve(L, self.y_train)
        mu = np.dot(Lk.T, Lf).reshape((n,))

        # Compute the covariance of the predictive posterior
        cov = K_ss - np.dot(Lk.T, Lk)

        return mu, cov

    def optimize_logLikelihood(self, param_bounds=np.array([[0, 0, 0], [10, 2, 2]])):
        """Optimize log likelihood to get the optimal hyperparameter of the kernel.

        Args:
            param_bounds (np.ndarray): lower and upper bounds for the BFGS optimization of
                the hyperparameter [[sigma_low, length_low, noise_low],
                                    [sigma_up, length_up, noise_up]]

        """
        initial_guess = [self.sigma, self.length, self.noise]
        bounds = opt.Bounds(lb=param_bounds[0, :], ub=param_bounds[1, :])

        res = opt.minimize(self.negativelogLikelihood, initial_guess,
                           args=(self.X, self.y_train), method="L-BFGS-B",
                           jac=self.gradient_negLogLikelihood, bounds=bounds,
                           options={'maxiter': 1000, 'gtol': 1e-6, 'disp': True})
        self.sigma = res.x[0]
        self.length = res.x[1]
        self.noise = res.x[2]

        print('The optimal hypervalues are sig={},'
              ' l={}, noise={}'.format(self.sigma, self.length, self.noise))
        return None

    def negativelogLikelihood(self, params: list, sqdist: np.ndarray, y: np.ndarray):
        """Obtain the negative log likelihood.

        Args:
            params (list): list of hyperparameter [sigma, length, noise]
            sqdist (np.ndarray): Euclidian Distance Matrix sqdist(x, x)
            y (np.ndarray): function output values corresponding to x

        Return:
            (np.ndarray): negative log likelihood

        """
        sigma = params[0]
        length = params[1]
        noise = params[2]
        m = np.shape(sqdist)[0]

        # sigma from training data
        cov_dat = np.exp(-sqdist + np.diag(self.eps*np.ones(m)))
        L = np.linalg.inv(np.linalg.cholesky(cov_dat))
        cov_dat_i = np.dot(L.T, L)
        sigma = float(np.dot(np.dot(y.T, cov_dat_i), y) / m)
        self.sigma = sigma

        # get inverse of K
        K = self.rbf_kernel(sqdist, [sigma, length]) + np.diag(noise**2 * np.ones(m))
        c = lin.inv(lin.cholesky(K + self.eps * np.eye(m)))
        Ki = np.dot(c.T, c)

        # log determinant of K
        (_, logdetK) = lin.slogdet(K)

        # likelihood
        ll = -m/2*np.log(2*np.pi) - logdetK/2 - 1/2*np.dot(y.T, np.dot(Ki, y))
        return -ll

    def gradient_negLogLikelihood(self, params: list, sqdist: np.ndarray, y: np.ndarray):
        """Calculate the gradient of negative log likelihood.

        Args:
            params (list): list of hyperparameter [sigma, length, noise]
            sqdist (np.ndarray): Euclidian Distance Matrix sqdist(x, x)
            y (np.ndarray): function output values corresponding to x

        Return:
            (np.ndarray): negative log likelihood

        """
        sigma = params[0]
        length = params[1]
        noise = params[2]
        m = np.shape(sqdist)[0]

        # get inverse of K
        K = self.rbf_kernel(sqdist, [sigma, length]) + np.diag(noise**2 * np.ones(m))
        c = lin.inv(lin.cholesky(K + self.eps * np.eye(m)))
        Ki = np.dot(c.T, c)

        # other useful variables
        KiY = np.dot(Ki, y)
        KsDist = np.dot(K, sqdist) / (length**2)

#        # gradients
#        dnll_length = (- 1/2 * np.sum(np.diag(np.dot(Ki, KsDist)))
#                       + m/2 * np.dot(KiY.T, np.dot(KsDist, KiY))/np.dot(y.T, KiY))
#        dnll_noise = (- 1/2 * np.sum(np.diag(Ki))
#                      + m/2 * np.dot(KiY.T, KiY)/np.dot(y.T, KiY))

        # gradient wrt sigma
        dnll_sigma = (m/sigma - 1/sigma * np.dot(KiY.T, np.dot(K, KiY)))

        # gradient wrt length
        dnll_length = (1/2 * np.sum(np.diag(np.dot(Ki, KsDist)))
                       - 1/2 * np.dot(KiY.T, np.dot(KsDist, KiY)))

        # gradient wrt noise
        dnll_noise = (noise * np.sum(np.diag(Ki))
                      - noise * np.dot(KiY.T, KiY))

        return np.concatenate((dnll_sigma, dnll_length, dnll_noise), axis=0)

    def gradient_checking(self, x, y, epsilon=1.0e-5):
        """Check the analytical gradients against the numerical ones.
 
        Args:

        """

        return None

    def sample_from_kernel(self, x_test: np.ndarray):
        """Sample curves from kernel and plot them.

        Args:
            x_test (np.ndarray)

        Return:
            f_prior (np.ndarray)

        """
        n = np.shape(x_test)[0]
        X_ss = self.square_distanceMatrix(x_test, x_test)
        K_ss = self.rbf_kernel(X_ss, [self.sigma, self.length])

        # Sample curves from kernel with Cholesky decomposition
        L = np.linalg.cholesky(K_ss + self.eps*np.eye(n))  # add small numbers for stability
        f_prior = np.dot(L, np.random.normal(0, 1, size=(n, 3)))

        # Plot sampling
        _, ax1 = plt.subplots()
        ax1.plot(x_test, f_prior, label='sample from kernel')
        plt.legend()

        return f_prior

    def sample_from_posterior(self, x_test: np.ndarray, axes=None, label='sim'):
        """Draw sample curves from the posterior.

        Args:
            x_test (np.ndarray)

        Return:
            f_post (np.ndarray)

        """
        n = np.shape(x_test)[0]
        (mu, cov) = self.predictive_posterior(x_test)
        stdv = np.sqrt(np.diag(cov))

        # Draw samples from the posterior at our test points.
        L = np.linalg.cholesky(cov + self.eps*np.eye(n))
        f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, 3)))

        # Plot results
        if axes is None:
            _, axes = plt.subplots()
        axes.plot(self.x_train, self.y_train, 'x', ms=8)
        axes.plot(x_test, f_post)
        plt.gca().fill_between(x_test.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
        axes.plot(x_test, mu, '--', lw=2, label=r'$\mu$ {}'.format(label))
        plt.title('The GP posterior using RBF-kernel')
        plt.legend(loc=0)

        return f_post
