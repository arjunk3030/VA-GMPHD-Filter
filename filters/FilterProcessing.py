from itertools import cycle
import matplotlib.pyplot as plt
import time
import logging
import numpy as np
from testing.ExperimentalResults import ObjectEvaluator
from PIL import Image
import filters.calculate_visibility as calculate_visibility
from util_files.logger_setup import logger
from filters.GMPHD.GMPHD import GMPHD, GaussianMixture
from filters.GMPHD.GMPHD_params import GMPHD_model
from filters.NOV_GMPHD.NOV_GMPHD import NOV_GMPHD, NOV_GaussianMixture
from filters.NOV_GMPHD.NOV_GMPHD_params import NOV_GMPHD_model
from filters.VA_GMPHD.VA_GMPHD import VA_GMPHD, VA_GaussianMixture
from filters.VA_GMPHD.VA_GMPHD_params import VA_GMPHD_model


class FilterProcessing:
    def __init__(self, ground_truth_objs, ground_truth_types, object_set, filter):
        self.ground_truth_objs = ground_truth_objs
        self.ground_truth_types = ground_truth_types
        self.object_set = object_set
        self.estimated_mean = []
        self.estimated_cls = []
        self.current_filter = filter
        if filter == "VA_GMPHD":
            self.model = VA_GMPHD_model()
            self.gmphd = VA_GMPHD(self.model)
            self.gaussian_mixture = VA_GaussianMixture([], [], [], [])
        elif filter == "NOV_GMPHD":
            self.model = NOV_GMPHD_model()
            self.gmphd = NOV_GMPHD(self.model)
            self.gaussian_mixture = NOV_GaussianMixture([], [], [], [])
        elif filter == "GMPHD":
            self.model = GMPHD_model()
            self.gmphd = GMPHD(self.model)
            self.gaussian_mixture = GaussianMixture([], [], [], [])

        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        logging.getLogger("pil").setLevel(logging.ERROR)
        self.all_measurements = []

    def run_filter(self, scene_pos, scene_ctrl, observed_means, observed_cls, distance, camera_width, camera_height):
        self.all_measurements.append(observed_means)
        a = time.time()

        v = self.gmphd.prediction(self.gaussian_mixture)
        p_v = [0] * len(v.w)
        if len(v.m) > 0 and self.current_filter != "GMPHD":
            p_v = calculate_visibility.calculate_all_p_v(
                self.object_set,
                scene_pos,
                scene_ctrl,
                v.m,
                v.cls,
                self.estimated_mean[-1],
                camera_width, 
                camera_height
                
            )
        v = self.gmphd.correction(v, p_v, observed_means, observed_cls, distance)
        self.gaussian_mixture = self.gmphd.pruning(v)
        estimated_mean, estimated_cls = self.gmphd.state_estimation(
            self.gaussian_mixture
        )

        if len(estimated_mean) == 0:
            self.estimated_mean.append([])
            self.estimated_cls.append([])
        else:
            combined = list(zip(list(estimated_cls), list(estimated_mean)))
            combined.sort(key=lambda x: x[0])
            sorted_cls, sorted_mean = zip(*combined)

            self.estimated_mean.append(list(sorted_mean))
            self.estimated_cls.append(list(sorted_cls))

            logger.info(f"{self.object_set}: {combined}")
        logger.info("Filtration time: " + str(time.time() - a) + " sec")

    def extract_axis_for_plot(self, X_collection, delta):
        time = []
        x = []
        y = []
        rot = []
        k = 0
        for X in X_collection:
            for state in X:
                x.append(state[0])
                y.append(state[1])
                rot.append(state[2])
                time.append(k)
            k += delta
        return time, x, y, rot

    def extract_ground_truths(self, X_collection, total_time):
        x = []
        y = []
        rot = []
        time = []
        for step in np.unique(np.array(total_time)):
            for X in X_collection:
                x.append(X[0])
                y.append(X[1])
                rot.append(X[2])
                time.append(step)
        return time, x, y, rot

    def plot_cardinality(self, estimated_counts, actual_count, total_time, time_step):
        plt.figure(figsize=(10, 6))
        times = [i * time_step for i in range(total_time)]

        plt.scatter(
            times, estimated_counts, color="blue", label="Estimated object count", s=20
        )
        plt.axhline(
            y=actual_count, color="r", linestyle="-", label="Actual object count"
        )

        plt.xlabel("Time (sec)")
        plt.ylabel("Count")
        plt.title("Estimated vs Actual Object Count")
        plt.legend()
        plt.grid(True)

        plt.savefig("testing/graphs/cardinality_plot.png")
        plt.close()

    def plot_3d(self, meas_time, meas_x, meas_y, estim_time, estim_x, estim_y):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(estim_time, estim_x, estim_y, c="green", label="Estimations", s=20)
        ax.scatter(
            meas_time, meas_x, meas_y, c="blue", label="Measurements", marker="x", s=15
        )

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("X (m)")
        ax.set_zlabel("Y (m)")
        ax.set_title("3D Visualization of Object Tracking")
        ax.legend()

        plt.savefig("testing/graphs/3d_plot.png")
        plt.close()

    def assign_colors_to_objects(self, num_objects):
        color_cycle = cycle(plt.cm.tab20(np.linspace(0, 1, 20)))
        return [next(color_cycle) for _ in range(num_objects)]

    def plot_2d(
        self,
        meas_time,
        meas_values,
        estim_time,
        estim_values,
        axis_label,
        title,
        file_name,
    ):
        plt.figure(figsize=(10, 6))

        # Determine the number of unique objects (assumed to be equal to the length of meas_values)
        times = np.unique(meas_time)

        # Assign unique colors to each object
        object_colors = self.assign_colors_to_objects(meas_time.count(meas_time[0]))

        # Add jitter to ground truth to show overlapping points
        jitter_amount = 0.009
        jittered_meas_values = [
            val + np.random.uniform(-jitter_amount, jitter_amount)
            for val in meas_values
        ]

        # Plot ground truth with different colors for each object
        previousTime = -1
        colorNumber = 0
        for idx, (time1, value) in enumerate(zip(meas_time, jittered_meas_values)):
            if time1 != previousTime:
                colorNumber = 0
                previousTime = time1
            plt.scatter(
                time1,
                value,
                c=[object_colors[colorNumber]],
                marker="x",
                s=85,  # Larger size for clarity
                label=None if idx > 0 else "Ground truth",  # Only label the first point
            )
            colorNumber += 1

        # Add jitter to estimated values as well
        jittered_estim_values = [
            val + np.random.uniform(-jitter_amount, jitter_amount)
            for val in estim_values
        ]

        # Plot estimated values (smaller gray circles)
        plt.scatter(
            estim_time,
            jittered_estim_values,
            c="gray",
            alpha=0.6,
            s=50,  # Smaller size for estimated points
            label="Filtered results",
        )
        # Set axis labels, title, and legend
        plt.xlabel("Time (sec)")
        plt.ylabel(axis_label)
        plt.title(title)
        plt.legend()

        # Add grid for better readability
        plt.grid(True)

        # Save the figure as a PNG file
        plt.savefig(f"testing/graphs/{file_name}.png")
        plt.close()

    def plot_misclassification_rate(self, misclassification_rates, total_time, T_s):
        import matplotlib.pyplot as plt

        time = [i * T_s for i in range(total_time)]
        plt.figure()
        plt.plot(time, misclassification_rates, label="Misclassification Rate")
        plt.xlabel("Time (s)")
        plt.ylabel("Misclassification Rate")
        plt.title("Misclassification Rate Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("testing/graphs/misclassification_rate.png")
        plt.close()

    def outputFilter(self):
        if len(self.estimated_mean) == 0:
            logger.info("No estimated states")
            return [], []

        meas_time, meas_x, meas_y, meas_rot = self.extract_axis_for_plot(
            self.all_measurements, self.model["T_s"]
        )
        estim_time, estim_x, estim_y, estim_rotation = self.extract_axis_for_plot(
            self.estimated_mean, self.model["T_s"]
        )

        # Cardinality plot
        estimated_counts = [len(X) for X in self.estimated_mean]
        actual_count = len(self.ground_truth_objs)
        total_time = len(self.estimated_mean)
        self.plot_cardinality(
            estimated_counts, actual_count, total_time, self.model["T_s"]
        )

        # 3D plot (no changes)
        self.plot_3d(meas_time, meas_x, meas_y, estim_time, estim_x, estim_y)

        # Extract ground truth values for X and Y from ground_truth_objs
        ground_truth_time, ground_truth_x, ground_truth_y, ground_truth_rot = (
            self.extract_ground_truths(self.ground_truth_objs, estim_time)
        )

        # 2D plots: Compare estimated values with ground truth (not measurements)
        self.plot_2d(
            ground_truth_time,
            ground_truth_x,
            estim_time,
            estim_x,
            "X (m)",
            "Time vs X coordinate",
            "x_plot",
        )
        self.plot_2d(
            ground_truth_time,
            ground_truth_y,
            estim_time,
            estim_y,
            "Y (m)",
            "Time vs Y coordinate",
            "y_plot",
        )

        self.plot_2d(
            ground_truth_time,
            ground_truth_rot,
            estim_time,
            estim_rotation,
            "Rotation (°)",
            "Time vs Rotation",
            "rotation_plot",
        )

        # Evaluate classification accuracy and store misclassification rate
        misclassification_rates = []
        for i in range(len(self.estimated_mean)):
            evaluator = ObjectEvaluator(
                self.estimated_mean[i],
                self.estimated_cls[i],
                self.ground_truth_objs,
                self.ground_truth_types,
            )
            correct_classifications, total_matched = evaluator.classify_accuracy()
            misclassification_rate = (
                1 - correct_classifications / total_matched if total_matched else 0
            )
            misclassification_rates.append(misclassification_rate)

        # Plot misclassification rate over time
        self.plot_misclassification_rate(
            misclassification_rates, total_time, self.model["T_s"]
        )

        # Display plots (optional)
        Image.open("testing/graphs/cardinality_plot.png").show()
        Image.open("testing/graphs/3d_plot.png").show()
        Image.open("testing/graphs/x_plot.png").show()
        Image.open("testing/graphs/y_plot.png").show()
        Image.open("testing/graphs/rotation_plot.png").show()
        Image.open("testing/graphs/misclassification_rate.png").show()

        return self.estimated_mean[-1], self.estimated_cls[-1]

    def evaluate(self):
        plt.rcParams["font.size"] = 12
        logger.info(f"\nEVALUATION for {self.object_set}:")
        logger.info("---------------------------------------")
        if len(self.estimated_mean) == 0:
            logger.info("No estimated states")
            return
        print(f"Here are the predicted object locations: {self.estimated_mean[-1]}\n")
        evaluator = ObjectEvaluator(
            self.estimated_mean[-1],
            self.estimated_cls[-1],
            self.ground_truth_objs,
            self.ground_truth_types,
        )

        filtered_count, ground_truth_count = evaluator.compare_object_count()
        logger.info(
            f"Filtered count is {filtered_count} and the ground truth count is {ground_truth_count}"
        )

        # Classification accuracy
        correct_classifications, total_matched = evaluator.classify_accuracy()
        classification_accuracy = (
            correct_classifications / total_matched if total_matched else 0
        )
        logger.info(f"Misclassification rate: {(1-classification_accuracy) * 100:.2f}%")

        # Distance error
        avg_dist_error = evaluator.calc_distance_error()
        logger.info(f"Average x and y pose error: {avg_dist_error:.2f}")

        # Pose error
        avg_pose_error = evaluator.calc_pose_error()
        logger.info(f"Average rotational pose error: {avg_pose_error:.2f}\n")
