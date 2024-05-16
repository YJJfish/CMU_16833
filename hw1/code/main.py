'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from ray_casting_result import RayCastingResult

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import cv2


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=1)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.5)
    scat.remove()

def visualize_timestep_opencv(occupancy_map_vis, X_bar, tstep, output_path):
    occupancy_map_vis_copy = occupancy_map_vis.copy()
    x_locs = (X_bar[:, 0] / 10.0).astype(np.int32)
    y_locs = 799 - (X_bar[:, 1] / 10.0).astype(np.int32)
    [cv2.circle(occupancy_map_vis_copy, [x_locs[i], y_locs[i]], radius=1, color=[0, 0, 255], thickness=cv2.FILLED) for i in range(x_locs.shape[0])]
    cv2.imwrite('{}/{:04d}.png'.format(output_path, tstep), occupancy_map_vis_copy)
    cv2.imshow("Visualize", occupancy_map_vis_copy)
    cv2.waitKey(100)


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))
    X_bar_init[:, 3] = 1.0 / num_particles

    batch_size = num_particles
    map_resolution = 10.0
    counter = 0
    while (counter < num_particles):
        y0_vals = np.random.uniform(0, 7000, (batch_size,))
        x0_vals = np.random.uniform(3000, 7000, (batch_size,))
        theta0_vals = np.random.uniform(-3.14159, 3.14159, (batch_size,))
        X, Y = (x0_vals/map_resolution).astype(np.int32), (y0_vals/map_resolution).astype(np.int32)
        valid = np.logical_and(occupancy_map[Y, X] >= 0.0, occupancy_map[Y, X] <= 0.05)
        x0_vals, y0_vals, theta0_vals = x0_vals[valid], y0_vals[valid], theta0_vals[valid]
        valid_num = x0_vals.shape[0]
        if counter + valid_num > num_particles:
            X_bar_init[counter:, 0] = x0_vals[0:num_particles-counter]
            X_bar_init[counter:, 1] = y0_vals[0:num_particles-counter]
            X_bar_init[counter:, 2] = theta0_vals[0:num_particles-counter]
        else:
            X_bar_init[counter:counter+valid_num, 0] = x0_vals
            X_bar_init[counter:counter+valid_num, 1] = y0_vals
            X_bar_init[counter:counter+valid_num, 2] = theta0_vals
        counter += valid_num

    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=5000, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--ray_casting_result_path', default='./raycasting.npz')
    args = parser.parse_args()
    args.visualize = True
    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    ray_casting_result = RayCastingResult()
    if os.path.exists(args.ray_casting_result_path):
        print("Find ray casting result file. Loading it ...")
        ray_casting_result.from_file(args.ray_casting_result_path)
        print("Successfully loaded ray casting result file.")
    else:
        print("Ray casting result file doesn't exist. Computing it ...")
        ray_casting_result.precompute(
            occupancy_map=occupancy_map,
            occupancy_threshold=0.35,
            map_resolution=10.0,
            subdivide=1,
            max_range=2000,
            visualize=False
        )
        ray_casting_result.to_file(args.ray_casting_result_path)
        print("Successfully wrote ray casting result file to disk.")
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(ray_casting_result)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)
        occupancy_map_vis = np.zeros(occupancy_map.shape, dtype=np.uint8)
        occupancy_map_vis[occupancy_map == -1] = 255
        occupancy_map_vis[occupancy_map >= 0] = ((1.0 - occupancy_map[occupancy_map >= 0]) / 1.5 * 255.0).astype(np.uint8)
        occupancy_map_vis = np.flipud(occupancy_map_vis)
        occupancy_map_vis = np.array([occupancy_map_vis]*3, dtype=np.uint8).transpose(1, 2, 0)
    
    first_time_idx = True
    in_room = 50
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            # visualize_timestep(X_bar, time_idx, args.output)
            visualize_timestep_opencv(occupancy_map_vis, X_bar, time_idx, args.output)