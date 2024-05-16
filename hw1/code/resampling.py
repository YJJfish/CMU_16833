'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        n = X_bar.shape[0]
        freqs = np.random.multinomial(n, X_bar[:, 3])
        X_bar_resampled = []
        for i, freq in enumerate(freqs):
            X_bar_resampled += [X_bar[i]] * freq
        return np.array(X_bar_resampled, dtype=np.float64)

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = np.empty_like(X_bar)
        M = X_bar.shape[0]
        weight = X_bar[:, 3] / np.sum(X_bar[:, 3])
        r = np.random.uniform(0, 1.0 / M)
        c = weight[0]
        i = 0
        for m in range(M):
            U = r + m / M
            while U > c:
                i += 1
                c += weight[i]
            X_bar_resampled[m] = X_bar[i]
        return X_bar_resampled