'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
from map_reader import MapReader
from matplotlib import pyplot as plt


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00005
        self._alpha2 = 0.00005
        self._alpha3 = 0.0075
        self._alpha4 = 0.0075

    @staticmethod
    def _sample_normal_distribution(b : float) -> float:
        b = math.sqrt(b)
        return 0.5 * np.random.uniform(-b, +b, (12,)).sum()
    
    @staticmethod
    def _sample_triangular_distribution(b : float) -> float:
        b = math.sqrt(b)
        return 6.0**0.5 / 2.0 * np.random.uniform(-b, +b, (2,)).sum()

    @staticmethod
    def _sample_numpy(b : float) -> float:
        b = math.sqrt(b)
        return np.random.normal(0, b)
    
    @staticmethod
    def _map_to_pi(angle : float) -> float:
        return angle - 2.0 * np.pi * np.floor((angle + np.pi) / (2.0 * np.pi))

    def update(self, u_t0 : np.ndarray, u_t1 : np.ndarray, x_t0 : np.ndarray) -> np.ndarray :
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        if (u_t0 == u_t1).all():
            return x_t0
        delta_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        # delta_rot1 = self._map_to_pi(delta_rot1)
        delta_trans = np.linalg.norm(u_t1[0:2] - u_t0[0:2])
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1
        # delta_rot2 = self._map_to_pi(delta_rot2)

        delta_rot1_hat = delta_rot1 - self._sample_numpy(self._alpha1 * delta_rot1**2 + self._alpha2 * delta_trans**2)
        # delta_rot1_hat = self._map_to_pi(delta_rot1_hat)
        delta_trans_hat = delta_trans - self._sample_numpy(self._alpha3 * delta_trans**2 + self._alpha4 * (delta_rot1**2 + delta_rot2**2))
        delta_rot2_hat = delta_rot2 - self._sample_numpy(self._alpha1 * delta_rot2**2 + self._alpha2 * delta_trans**2)
        # delta_rot2_hat = self._map_to_pi(delta_rot2_hat)
        return \
            np.array([
                x_t0[0] + delta_trans_hat * math.cos(x_t0[2] + delta_rot1_hat),
                x_t0[1] + delta_trans_hat * math.sin(x_t0[2] + delta_rot1_hat),
                self._map_to_pi(x_t0[2] + delta_rot1_hat + delta_rot2_hat)
            ], dtype=np.float64)
    