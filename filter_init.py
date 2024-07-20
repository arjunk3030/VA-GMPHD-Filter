import matplotlib.pyplot as plt
import time
from Constants import TOTAL_STEPS
import json
import numpy as np
from gmphd import GaussianMixture, GmphdFilter, clutter_intensity_function
from mpl_toolkits.mplot3d import Axes3D


def smaller_grid_model(total_frames):
    model = {}

    # Sampling time, time step duration
    T_s = 1.0
    model["T_s"] = T_s

    # number of scans, number of iterations in our simulation
    model["num_scans"] = total_frames

    # Surveillance region
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model["p_s"] = 1.05

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model["F"] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s**4) / 4 * I_2
    Q[0:2, 2:] = (T_s**3) / 2 * I_2
    Q[2:, 0:2] = (T_s**3) / 2 * I_2
    Q[2:, 2:] = (T_s**2) * I_2
    # standard deviation of the process noise
    sigma_w = 0.1
    Q = Q * (sigma_w**2)
    model["Q"] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model["F_spawn"] = []
    model["d_spawn"] = []
    model["Q_spawn"] = []
    model["w_spawn"] = []

    # Parameters of the new born targets Gaussian mixture
    w = [0.05] * 2
    m = [np.array([2, -0.1, 0.0, 0.0]), np.array([-0.45, -0.1, 0.0, 0.0])]
    P_pom_ = np.diag([4, 4, 0.1, 0.1])
    P = [P_pom_.copy(), P_pom_.copy()]
    model["birth_GM"] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model["p_d"] = 0.14

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model["H"] = np.zeros((2, 4))
    model["H"][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 0.1  # m
    model["R"] = I_2 * (sigma_v**2)

    # the reference to clutter intensity function
    model["lc"] = 0.0
    model["clutt_int_fun"] = lambda z: clutter_intensity_function(
        z, model["lc"], model["surveillance_region"]
    )

    # pruning and merging parameters:
    model["T"] = 0.1
    model["U"] = 0.2
    model["Jmax"] = 20

    return model


def two_cups_model(total_frames):
    model = {}

    # Sampling time, time step duration
    T_s = 1.0
    model["T_s"] = T_s

    # number of scans, number of iterations in our simulation
    model["num_scans"] = total_frames

    # Surveillance region
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model["p_s"] = 1

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model["F"] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s**4) / 4 * I_2
    Q[0:2, 2:] = (T_s**3) / 2 * I_2
    Q[2:, 0:2] = (T_s**3) / 2 * I_2
    Q[2:, 2:] = (T_s**2) * I_2
    # standard deviation of the process noise
    sigma_w = 0.05
    Q = Q * (sigma_w**2)
    model["Q"] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model["F_spawn"] = []
    model["d_spawn"] = []
    model["Q_spawn"] = []
    model["w_spawn"] = []

    # Parameters of the new born targets Gaussian mixture
    w = [0.01] * 2
    m = [np.array([2, -0.1, 0.0, 0.0]), np.array([-0.45, -0.1, 0.0, 0.0])]
    P_pom_ = np.diag([4, 4, 0.1, 0.1])
    P = [P_pom_.copy(), P_pom_.copy()]
    model["birth_GM"] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model["p_d"] = 0.14

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model["H"] = np.zeros((2, 4))
    model["H"][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 0.1  # m
    model["R"] = I_2 * (sigma_v**2)

    # the reference to clutter intensity function
    model["lc"] = 0
    model["clutt_int_fun"] = lambda z: clutter_intensity_function(
        z, model["lc"], model["surveillance_region"]
    )

    # pruning and merging parameters:
    model["T"] = 0.2
    model["U"] = 0.3
    model["Jmax"] = 100

    return model

    # This is the model for the example in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark
    # The implementation almost analog to Matlab code provided by Vo in http://ba-tuong.vo-au.com/codes.html

    model = {}

    # Sampling time, time step duration
    T_s = 1.0
    model["T_s"] = T_s

    # number of scans, number of iterations in our simulation
    model["num_scans"] = 100

    # Surveillance region
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000
    model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model["p_s"] = 1

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model["F"] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s**4) / 4 * I_2
    Q[0:2, 2:] = (T_s**3) / 2 * I_2
    Q[2:, 0:2] = (T_s**3) / 2 * I_2
    Q[2:, 2:] = (T_s**2) * I_2
    # standard deviation of the process noise
    sigma_w = 5.0
    Q = Q * (sigma_w**2)
    model["Q"] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model["F_spawn"] = []
    model["d_spawn"] = []
    model["Q_spawn"] = []
    model["w_spawn"] = []

    # Parameters of the new born targets Gaussian mixture
    w = [0.1, 0.1]
    m = [np.array([500.0, 500.0, 0.0, 0.0]), np.array([700.0, 700.0, 0.0, 0.0])]
    P_pom_ = np.diag([500, 500, 1, 1])
    P = [P_pom_.copy(), P_pom_.copy()]
    model["birth_GM"] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model["p_d"] = 1

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model["H"] = np.zeros((2, 4))
    model["H"][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 10  # m
    model["R"] = I_2 * (sigma_v**2)

    # the reference to clutter intensity function
    model["lc"] = 50
    model["clutt_int_fun"] = lambda z: clutter_intensity_function(
        z, model["lc"], model["surveillance_region"]
    )

    # pruning and merging parameters:
    model["T"] = 1e-5
    model["U"] = 4.0
    model["Jmax"] = 100

    return model


def true_trajectory_tracks_plots(targets_birth_time, targets_tracks, delta):
    for_plot = {}
    for i, birth in enumerate(targets_birth_time):
        brojac = birth
        x = []
        y = []
        time = []
        for state in targets_tracks[i]:
            x.append(state[0])
            y.append(state[1])
            time.append(brojac)
            brojac += delta
        for_plot[i] = (time, x, y)
    return for_plot


def extract_axis_for_plot(X_collection, delta):
    time = []
    x = []
    y = []
    k = 0
    for X in X_collection:
        for state in X:
            x.append(state[0])
            y.append(state[1])
            time.append(k)
        k += delta
    return time, x, y


def run_filter(measurements):
    measurements = [
        [np.array(sublist) for sublist in sublist_list] for sublist_list in measurements
    ]
    targets_birth_time = []
    targets_tracks = []

    model = two_cups_model(TOTAL_STEPS)
    # Call of the gmphd filter for the created observations collections
    gmphd = GmphdFilter(model)
    a = time.time()
    X_collection = gmphd.filter_data(measurements)
    print("Filtration time: " + str(time.time() - a) + " sec")

    # Plot the results of filtration saved in X_collection file
    tracks_plot = true_trajectory_tracks_plots(
        targets_birth_time, targets_tracks, model["T_s"]
    )

    # True ones
    points = [[-0.5, 0], [0.5, 0]]
    # Plot measurements, true trajectories and estimations
    meas_time, meas_x, meas_y = extract_axis_for_plot(measurements, model["T_s"])
    estim_time, estim_x, estim_y = extract_axis_for_plot(X_collection, model["T_s"])

    plt.figure()
    plt.plot(meas_time, meas_x, "x", c="C0")
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, x, "r")
    plt.plot(estim_time, estim_x, "o", c="k", markersize=3)
    plt.xlabel("time[$sec$]")
    plt.ylabel("x")
    plt.title(
        "X axis in time. Blue x are measurements(50 in each time step), "
        "black dots are estimations and the red lines are actual trajectories of targets",
        loc="center",
        wrap=True,
    )

    plt.figure()
    plt.plot(meas_time, meas_y, "x", c="C0")
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, y, "r")
    plt.plot(estim_time, estim_y, "o", c="k", markersize=3)
    plt.xlabel("time[$sec$]")
    plt.ylabel("y")
    plt.title(
        "Y axis in time. Blue x are measurements(50 in each time step), "
        "black dots are estimations and the red lines are actual trajectories of targets",
        loc="center",
        wrap=True,
    )

    x_list = [point[0] for point in points]
    true_x = [x_list[:] for _ in range(TOTAL_STEPS)]

    y_list = [point[1] for point in points]
    true_y = [y_list[:] for _ in range(TOTAL_STEPS)]

    true_times = [i for i in range(0, TOTAL_STEPS) for _ in range(len(points))]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(estim_time, estim_y, estim_x, color="green", label="Projections")
    # ax.scatter(true_times, true_y, true_x, color="red", label="Truth")
    ax.scatter(
        meas_time, meas_y, meas_x, color="blue", label="Measurements", marker="x"
    )

    # setting title and labels
    ax.set_title("3D plot")
    ax.set_xlabel("Time")
    ax.set_ylabel("X axis")
    ax.set_zlabel("Y axis")

    def on_legend_click(event):
        legend = event.artist
        for handle in legend.legendHandles:
            handle.set_visible(not handle.get_visible())
        plt.draw()

    fig.canvas.mpl_connect("pick_event", on_legend_click)

    ax.legend()

    # displaying the plot
    # plt.show()

    num_targets_truth = []
    num_targets_estimated = []

    for x_set in X_collection:
        num_targets_estimated.append(len(x_set))

    plt.figure()
    (markerline, stemlines, baseline) = plt.stem(
        num_targets_estimated, label="estimated number of targets"
    )
    plt.setp(baseline, color="k")  # visible=False)
    plt.setp(stemlines, visible=False)  # visible=False)
    plt.setp(markerline, markersize=3.0)
    plt.step(num_targets_truth, "r", label="actual number of targets")
    plt.xlabel("time[$sec$]")
    plt.legend()
    plt.title("Estimated cardinality VS actual cardinality", loc="center", wrap=True)
    plt.show()


if __name__ == "__main__":
    file_name = f"data.json"
    with open(file_name, "r") as f:
        data = json.load(f)

    # Retrieve the lists from the dictionary
    measurements = data["measurements"]

    run_filter(measurements)
