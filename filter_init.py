import matplotlib.pyplot as plt
import time
from Constants import TOTAL_STEPS
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gmphd_copy import GmphdFilter, GaussianMixture, clutter_intensity_function
from PIL import Image
import PointEstimation


class PhdFilter:
    def __init__(self):
        self.gaussian_mixture = GaussianMixture([], [], [])
        self.estimated_states = []
        self.model = self.two_cups_model()
        self.gmphd = GmphdFilter(self.model)
        self.all_measurements = []

    def two_cups_model(self):
        model = {}

        # Sampling time, time step duration
        T_s = 1.0
        model["T_s"] = T_s

        # Surveillance region
        x_min = -2
        x_max = 2
        y_min = -2
        y_max = 2
        model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

        # TRANSITION MODEL
        # Probability of survival (Commented out since it's not used)
        # model["p_s"] = 1

        # Transition matrix
        I_2 = np.eye(2)
        # F = [[I_2, T_s*I_2], [02, I_2]
        F = I_2  # Since we are only dealing with (x, y) and not velocities
        model["F"] = F

        # Process noise covariance matrix
        Q = (T_s**2) * I_2
        # Standard deviation of the process noise
        sigma_w = 0.1
        Q = Q * (sigma_w**2)
        model["Q"] = Q

        # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
        model["F_spawn"] = []
        model["d_spawn"] = []
        model["Q_spawn"] = []
        model["w_spawn"] = []

        # MEASUREMENT MODEL
        # Probability of detection
        model["p_d"] = 0.5

        # Measurement matrix z = Hx + v = N(z; Hx, R)
        model["H"] = I_2  # Since we are only measuring (x, y)
        # Measurement noise covariance matrix
        sigma_v = 0.1  # m
        model["R"] = I_2 * (sigma_v**2)

        # The reference to clutter intensity function
        model["lc"] = 0.1
        model["clutt_int_fun"] = lambda z: clutter_intensity_function(
            z, model["lc"], model["surveillance_region"]
        )

        # Pruning and merging parameters
        model["T"] = 0.15
        model["U"] = 0.4
        model["Jmax"] = 100

        return model

    def extract_axis_for_plot(self, X_collection, delta):
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

    def run_filter(self, measurements, depth_img, intrinsics, extrinsics):
        self.all_measurements.append(measurements)
        a = time.time()

        v = self.gmphd.prediction(self.gaussian_mixture)
        visibilities = []
        for means in v.m:
            visibilities.append(
                PointEstimation.is_point_range_visible(
                    np.array([means[0], means[1], 0.5]),
                    depth_img,
                    intrinsics,
                    extrinsics,
                )
            )
        v = self.gmphd.correction(v, visibilities, measurements)
        self.gaussian_mixture = self.gmphd.pruning(v)
        estimation = self.gmphd.state_estimation(self.gaussian_mixture)
        self.estimated_states.append(estimation)
        print(estimation)

        print("Filtration time: " + str(time.time() - a) + " sec")

    def outputFilter(self):
        if len(self.estimated_states) == 0:
            print("no estimated states")
            return
        tracks_plot = []

        # True ones
        points = [[-0.75, 0.5], [0, 0.5], [0.75, 0.5], [-0.75, 0], [0.1, -0.2]]
        # Plot measurements, true trajectories and estimations
        meas_time, meas_x, meas_y = self.extract_axis_for_plot(
            self.all_measurements, self.model["T_s"]
        )
        estim_time, estim_x, estim_y = self.extract_axis_for_plot(
            self.estimated_states, self.model["T_s"]
        )

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
        ax.scatter(true_times, true_y, true_x, color="red", label="Truth")
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
        image_path = "ploot.png"
        plt.savefig(image_path)

        # Load the saved image using PIL
        pil_image = Image.open(image_path)

        pil_image.show()

        # Display the plot
        # plt.show()

        num_targets_truth = []
        num_targets_estimated = []

        for x_set in self.estimated_states:
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
        plt.title(
            "Estimated cardinality VS actual cardinality", loc="center", wrap=True
        )
        image_path = "plot_sin_x.png"
        plt.savefig(image_path)

        # Load the saved image using PIL
        pil_image = Image.open(image_path)

        pil_image.show()
        plt.show()

        return self.estimated_states
