import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_log', default='../data/log/robotdata2.log')
args = parser.parse_args()
logfile = open(args.path_to_log, 'r')
for time_idx, line in enumerate(logfile):
    meas_type = line[0]
    meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
    odometry_robot = meas_vals[0:3]
    time_stamp = meas_vals[-1]
    if (meas_type == "L"):
        odometry_laser = meas_vals[3:6]
        ranges = meas_vals[6:-1]
        map_vis = np.zeros((800, 800, 3), dtype=np.uint8)
        ray_origin = np.array([4000.0, 4000.0], dtype=np.float64)
        ray_dir = meas_vals[2] - np.pi / 2.0 + (179.5 - np.arange(180)) / 180.0 * np.pi
        ray_dir = np.hstack([np.cos(ray_dir).reshape(180, 1), np.sin(ray_dir).reshape(180, 1)])
        for i in range(180):
            ray_target = ray_origin + ranges[i] * ray_dir[i]
            cv2.line(map_vis, (ray_origin/10).astype(np.int32), (ray_target/10).astype(np.int32),[255,255,255],thickness=1)
        cv2.line(map_vis, (ray_origin/10).astype(np.int32), ((ray_origin+100.0*ray_dir[90])/10).astype(np.int32),[255,0,0],thickness=2)
        cv2.imshow("vis", map_vis)
        key = cv2.waitKey(20)
        if key == 27:
            break