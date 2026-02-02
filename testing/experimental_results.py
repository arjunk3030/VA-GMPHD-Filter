import numpy as np

from util_files.object_parameters import ROTATION_INVARIANT


class ObjectEvaluator:
    def __init__(
        self,
        filtered_objs,
        filtered_types,
        ground_truth_objs,
        ground_truth_types,
        distance_threshold=0.3,
    ):
        self.filtered_objs = np.array(filtered_objs)
        self.filtered_types = filtered_types
        self.ground_truth_objs = np.array(ground_truth_objs)
        self.ground_truth_types = ground_truth_types
        self.distance_threshold = distance_threshold
        self.matched_pairs = []

        self._match_points()

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _match_points(self):
        for i, filt_obj in enumerate(self.filtered_objs):
            min_dist = float("inf")
            matched_idx = None

            for j, gt_obj in enumerate(self.ground_truth_objs):
                if any(match[1] == j for match in self.matched_pairs):
                    continue

                dist = self.euclidean_distance(filt_obj[:2], gt_obj[:2])
                if dist < min_dist:
                    min_dist = dist
                    matched_idx = j

            if matched_idx is not None and min_dist < self.distance_threshold:
                self.matched_pairs.append((i, matched_idx))

    def compare_object_count(self):
        return len(self.filtered_objs), len(self.ground_truth_objs)

    def classify_accuracy(self):
        correct_classifications = 0
        for filt_idx, gt_idx in self.matched_pairs:
            if self.filtered_types[filt_idx] == self.ground_truth_types[gt_idx]:
                correct_classifications += 1
        return correct_classifications, len(self.matched_pairs)

    def calc_distance_error(self):
        total_dist_error = 0
        for filt_idx, gt_idx in self.matched_pairs:
            dist = self.euclidean_distance(
                self.filtered_objs[filt_idx][:2], self.ground_truth_objs[gt_idx][:2]
            )
            total_dist_error += dist

        avg_dist_error = (
            total_dist_error / len(self.matched_pairs) if self.matched_pairs else 0
        )
        return avg_dist_error

    def calc_pose_error(self):
        total_pose_error = 0
        relevant_count = 0

        for filt_idx, gt_idx in self.matched_pairs:
            if self.filtered_types[filt_idx] not in ROTATION_INVARIANT:
                pose_error = abs(
                    self.filtered_objs[filt_idx][2] - self.ground_truth_objs[gt_idx][2]
                )
                total_pose_error += pose_error
                relevant_count += 1

        avg_pose_error = total_pose_error / relevant_count if relevant_count > 0 else 0
        return avg_pose_error
