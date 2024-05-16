'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from ray_casting_result import RayCastingResult


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, ray_casting_result : RayCastingResult):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 10
        self._z_short = 0.09
        self._z_max = 0.05
        self._z_rand = 2000

        self._sigma_hit = 100
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = ray_casting_result.max_range

        self.ray_casting_result = ray_casting_result

    def beam_range_finder_model(self, z_t1, x_t1):
        """
        param[in] z_t1 : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # ray casting
        laser_pos_x = x_t1[0] + np.cos(x_t1[2])*25
        laser_pos_y = x_t1[1] + np.sin(x_t1[2])*25
        z_t1_gt = self.ray_casting_result.get_180(laser_pos_x, laser_pos_y, x_t1[2])
        # compute probability
        prob_hit = np.where(np.logical_and(z_t1 >= 0.0, z_t1 <= self._max_range),
                            norm.pdf(z_t1, loc=z_t1_gt, scale=self._sigma_hit) / (norm.cdf(self._max_range,  loc=z_t1_gt, scale=self._sigma_hit) - norm.cdf(0.0,  loc=z_t1_gt, scale=self._sigma_hit)),
                            0.0
                            )
        prob_short = np.where(np.logical_and(z_t1 >= 0.0, z_t1 <= z_t1_gt),
                              self._lambda_short * np.exp(-self._lambda_short * z_t1) / (1.0 - np.exp(-self._lambda_short * z_t1_gt)),
                              0.0
                              )
        prob_max = np.where(z_t1 >= self._max_range,
                            1.0,
                            0.0
                            )
        prob_rand = np.where(np.logical_and(z_t1 >= 0.0, z_t1 < self._max_range),
                             1.0 / self._max_range,
                             0.0
                             )
        prob_zt1 = self._z_hit * prob_hit + \
                   self._z_short * prob_short + \
                   self._z_max * prob_max + \
                   self._z_rand * prob_rand
        prob_zt1 = prob_zt1[::2]
        prob_zt1 = np.delete(prob_zt1, np.where(prob_zt1 < 1e-10))
        # res = np.prod(prob_zt1)
        res = np.exp(np.sum(np.log(prob_zt1)))
        return res
