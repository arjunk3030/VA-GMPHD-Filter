import matplotlib.pyplot as plt
import time
from Constants import TOTAL_STEPS
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gmphd_copy import GmphdFilter, GaussianMixture, clutter_intensity_function
from PIL import Image
import PHDFilterCalculations
import Constants
from scipy.optimize import linear_sum_assignment


class PhdFilter:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        self.gaussian_mixture = GaussianMixture([], [], [], [])
        self.estimated_mean = []
        self.estimated_cls = []
        self.model = self.exp_2_model()
        self.gmphd = GmphdFilter(self.model)
        self.all_measurements = []

    def exp_2_model(self):
        model = {}

        # Sampling time, time step duration
        T_s = 1.0
        model["T_s"] = T_s
        model["nObj"] = Constants.YCB_OBJECT_COUNT

        # Surveillance region
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

        # TRANSITION MODEL
        # Probability of survival (Commented out since it's not used)
        # model["p_s"] = 1

        # Transition matrix
        I_3 = np.eye(3)

        # Process noise covariance matrix
        Q = (T_s**2) * I_3
        # Standard deviation of the process noise
        sigma_w_xy = 0.03  # Standard deviation for x and y
        sigma_w_z = 60
        Q = np.diag([sigma_w_xy**2, sigma_w_xy**2, sigma_w_z**2])
        model["Q"] = Q

        model["birth_w"] = 0.6
        model["birth_P"] = np.diag([0.0375, 0.0375, 50])

        # MEASUREMENT MODEL
        # Probability of detection
        model["p_d"] = 0.5
        model["alpha"] = 1

        # Measurement matrix z = Hx + v = N(z; Hx, R)
        model["H"] = I_3  # Since we are now measuring (x, y, z)
        # Measurement noise covariance matrix
        sigma_v_xy = 0.05  # Standard deviation for measurement noise in x and y
        sigma_v_z = (
            50  # Larger standard deviation for z due to higher measurement noise
        )
        model["R"] = np.diag([sigma_v_xy**2, sigma_v_xy**2, sigma_v_z**2])

        # The reference to clutter intensity function
        model["lc"] = 0.05
        model["clutt_int_fun"] = lambda z: clutter_intensity_function(
            z, model["lc"], model["surveillance_region"]
        )

        model["A"] = 0.16
        # Pruning and merging parameters
        model["T"] = 0.2
        model["U"] = 0.09
        model["Jmax"] = 60

        return model

    def exp_1_model(self):
        model = {}

        # Sampling time, time step duration
        T_s = 1.0
        model["T_s"] = T_s
        model["nObj"] = Constants.YCB_OBJECT_COUNT

        # Surveillance region
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

        # TRANSITION MODEL
        # Probability of survival (Commented out since it's not used)
        # model["p_s"] = 1

        # Transition matrix
        I_3 = np.eye(3)

        # Process noise covariance matrix
        Q = (T_s**2) * I_3
        # Standard deviation of the process noise
        sigma_w_xy = 0.05  # Standard deviation for x and y
        sigma_w_z = 50
        Q = np.diag([sigma_w_xy**2, sigma_w_xy**2, sigma_w_z**2])
        model["Q"] = Q

        model["birth_w"] = 0.5
        model["birth_P"] = np.diag([0.0375, 0.0375, 50])

        # MEASUREMENT MODEL
        # Probability of detection
        model["p_d"] = 0.65
        model["alpha"] = 1.25

        # Measurement matrix z = Hx + v = N(z; Hx, R)
        model["H"] = I_3  # Since we are now measuring (x, y, z)
        # Measurement noise covariance matrix
        sigma_v_xy = 0.04  # Standard deviation for measurement noise in x and y
        sigma_v_z = (
            50  # Larger standard deviation for z due to higher measurement noise
        )
        model["R"] = np.diag([sigma_v_xy**2, sigma_v_xy**2, sigma_v_z**2])

        # The reference to clutter intensity function
        model["lc"] = 0.06
        model["clutt_int_fun"] = lambda z: clutter_intensity_function(
            z, model["lc"], model["surveillance_region"]
        )

        model["A"] = 0.18
        # Pruning and merging parameters
        model["T"] = 0.15
        model["U"] = 0.06
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

    def run_filter(self, scene_pos, observed_means, observed_cls):
        self.all_measurements.append(observed_means)
        a = time.time()

        v = self.gmphd.prediction(self.gaussian_mixture)
        p_v = [0] * len(v.w)
        if len(v.m) > 0:
            p_v = PHDFilterCalculations.calculate_all_p_v(
                scene_pos,
                v.m,
                v.cls,
                self.estimated_mean[-1],
            )
        v = self.gmphd.correction(v, p_v, observed_means, observed_cls)
        self.gaussian_mixture = self.gmphd.pruning(v)
        estimated_mean, estimated_cls = self.gmphd.state_estimation(
            self.gaussian_mixture
        )
        # self.estimated_mean.append(estimated_mean)
        # self.estimated_cls.append(estimated_cls)

        if len(estimated_mean) == 0:
            self.estimated_mean.append([])
            self.estimated_cls.append([])
        else:
            combined = list(zip(list(estimated_cls), list(estimated_mean)))
            combined.sort(key=lambda x: x[0])
            sorted_cls, sorted_mean = zip(*combined)

            self.estimated_mean.append(list(sorted_mean))
            self.estimated_cls.append(list(sorted_cls))

        print(combined)
        print("Filtration time: " + str(time.time() - a) + " sec")

    def calculate_differences(ground_truth, observed, penalty_for_unmatched=None):
        # Create a distance matrix between ground truth and observed points
        distance_matrix = np.linalg.norm(ground_truth[:, np.newaxis] - observed, axis=2)

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Calculate the differences for matched points
        matched_differences = ground_truth[row_ind] - observed[col_ind]

        unmatched_ground_truth = np.setdiff1d(np.arange(len(ground_truth)), row_ind)
        unmatched_observed = np.setdiff1d(np.arange(len(observed)), col_ind)

        if penalty_for_unmatched is not None:
            # Apply penalties for unmatched points
            unmatched_differences = []
            for i in unmatched_ground_truth:
                unmatched_differences.append(
                    np.full_like(ground_truth[i], penalty_for_unmatched)
                )

            for i in unmatched_observed:
                unmatched_differences.append(
                    np.full_like(observed[i], penalty_for_unmatched)
                )

            unmatched_differences = np.array(unmatched_differences)
        else:
            unmatched_differences = np.array([]).reshape(0, ground_truth.shape[1])

        # Combine matched and unmatched differences
        all_differences = np.vstack([matched_differences, unmatched_differences])

        return (
            all_differences,
            row_ind,
            col_ind,
            unmatched_ground_truth,
            unmatched_observed,
        )

    def outputFilter(self):
        if len(self.estimated_mean) == 0:
            print("no estimated states")
            return
        tracks_plot = []

        # Plot measurements, true trajectories and estimations
        meas_time, meas_x, meas_y = self.extract_axis_for_plot(
            self.all_measurements, self.model["T_s"]
        )
        estim_time, estim_x, estim_y = self.extract_axis_for_plot(
            self.estimated_mean, self.model["T_s"]
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

        # x_list = [object[0][0] for object in self.ground_truth]
        # true_x = [x_list[:] for _ in range(TOTAL_STEPS)]

        # y_list = [point[0][1] for point in self.ground_truth]
        # true_y = [y_list[:] for _ in range(TOTAL_STEPS)]

        true_times = [
            i for i in range(0, TOTAL_STEPS) for _ in range(len(self.ground_truth))
        ]

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
        image_path = "plot.png"
        plt.savefig(image_path)

        # Load the saved image using PIL
        pil_image = Image.open(image_path)

        pil_image.show()

        # Display the plot
        # plt.show()

        num_targets_truth = []
        num_targets_estimated = []

        for x_set in self.estimated_mean:
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

        # print(f"Ground turth is {self.ground_truth}")
        # differences, row_ind, col_ind, unmatched_gt, unmatched_obs = (
        #     self.calculate_differences(
        #         np.array(self.ground_truth), np.array(self.estimated_mean[-1])
        #     )
        # )
        # print("Differences:", differences)
        return self.estimated_mean[-1], self.estimated_cls[-1]
