'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.linalg
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    return angle_rad - np.floor((angle_rad + np.pi) / (2.0 * np.pi)) * (2.0 * np.pi)


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2
    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    x, y, theta = init_pose[0, 0], init_pose[1, 0], init_pose[2, 0]
    # landmark[0::2] = x + init_measure[1::2] * np.cos(theta + init_measure[0::2])
    # landmark[1::2] = y + init_measure[1::2] * np.sin(theta + init_measure[0::2])
    for i in range(k):
        beta = init_measure[i*2+0,0]
        l = init_measure[i*2+1,0]
        landmark[i*2+0] = x + l * np.cos(theta + beta)
        landmark[i*2+1] = y + l * np.sin(theta + beta)
        comb_conv = scipy.linalg.block_diag(init_pose_cov, init_measure_cov)
        G = np.array([
            [1.0, 0.0, -l*np.sin(theta+beta), -l*np.sin(theta+beta), np.cos(theta+beta)],
            [0.0, 1.0, l*np.cos(theta+beta), l*np.cos(theta+beta), np.sin(theta+beta)]
        ], dtype=np.float64)
        landmark_cov[i*2:i*2+2, i*2:i*2+2] = G @ comb_conv @ G.T
    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    assert(3+2*k == X.shape[0])
    x, y, theta = X[0, 0], X[1, 0], X[2, 0]
    d, alpha = control[0, 0], control[1, 0]

    # State
    X_pre = np.zeros((3+2*k, 1), dtype=np.float64)
    X_pre[0, 0] = x + d * np.cos(theta)
    X_pre[1, 0] = y + d * np.sin(theta)
    X_pre[2, 0] = theta + alpha
    X_pre[3:3+2*k, 0] = X[3:3+2*k, 0]

    # Conv
    comb_conv = scipy.linalg.block_diag(control_cov, P)
    G = np.zeros((3+2*k, 6+2*k), dtype=np.float64)
    G[0:3, 0:3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    G[0:3, 3:6] = np.array([
        [1.0, 0.0, -d * np.sin(theta)],
        [0.0, 1.0, d * np.cos(theta)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    G[3:3+2*k, 6:6+2*k] = np.eye(2*k, dtype=np.float64)
    P_pre = G @ comb_conv @ G.T
    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    assert(3+2*k == X_pre.shape[0])
    for i in range(k):
        x_pre, y_pre, theta_pre = X_pre[0, 0], X_pre[1, 0], X_pre[2, 0]
        lx_pre = X_pre[3+2*i+0, 0]
        ly_pre = X_pre[3+2*i+1, 0]
        delta = np.array([lx_pre - x_pre, ly_pre - y_pre], dtype=np.float64)
        q = np.dot(delta, delta)
        measure_pre = np.zeros((2,1), dtype=np.float64) # \hat{z}_t^i
        measure_pre[0, 0] = warp2pi(np.arctan2(delta[1], delta[0]) - theta_pre)
        measure_pre[1, 0] = np.sqrt(q)
        H = np.zeros((2, 3+2*k), dtype=np.float64)
        H[0, 0] = delta[1] / q
        H[1, 0] = -delta[0] / np.sqrt(q)
        H[0, 1] = -delta[0] / q
        H[1, 1] = -delta[1] / np.sqrt(q)
        H[0, 2] = -1.0
        H[0, 3+i*2+0] = -delta[1] / q
        H[1, 3+i*2+0] = delta[0] / np.sqrt(q)
        H[0, 3+i*2+1] = delta[0] / q
        H[1, 3+i*2+1] = delta[1] / np.sqrt(q)
        K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + measure_cov) # 3+2*k x 2
        X_pre = X_pre + K @ (measure[i*2:i*2+2, :] - measure_pre)
        P_pre = (np.eye(3+2*k,dtype=np.float64) - K @ H) @ P_pre
    return X_pre, P_pre


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=np.float64)

    euclidean = np.linalg.norm(X[3:, 0].reshape(k, 2) - l_true.reshape(k, 2), axis=1)
    print("Euclidean: ", euclidean)
    mahalanobis = np.zeros((k,), dtype=np.float64)
    for i in range(k):
        delta = X[3+i*2:3+i*2+2, :] - l_true[i*2:i*2+2][:, np.newaxis]
        mahalanobis[i] = np.sqrt(delta.T @ np.linalg.inv(P[3+i*2:3+i*2+2, 3+i*2:3+i*2+2]) @ delta)
    print("Mahalanobis: ", mahalanobis)

    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)
    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)

if __name__ == "__main__":
    main()
