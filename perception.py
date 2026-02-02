import logging
import numpy as np
import time
import math
from util_files.TrajectorySettings import View
import PointEstimation
from object_detection import detect_objects
from DenseProcessor import DenseProcessor
from util_files.util import camera_intrinsic
    
# Helper to step simulation
def step_sim(model, data, viewer, dt=0.01):
    import mujoco
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
    mujoco.mj_forward(model, data)


class PerceptionManager:
    def __init__(self, model, data, viewer, renderer, depth_renderer, camera_name, camera_height, camera_width, threshold=1.0, object_detection_model=None, ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.r = renderer
        self.dr = depth_renderer
        self.camera_name = camera_name
        self.threshold = threshold
        self.object_detection_model = object_detection_model
        self.camera_height = camera_height
        self.camera_width = camera_width

    def capture_view(self, object_set_name: str, step_number: int) -> View:
        """Capture RGB+depth and optionally detect objects, returning a View dataclass."""
        try:
            target_body_id = self.model.body(object_set_name).id
            self.model.camera(self.camera_name).targetbodyid = target_body_id
        except Exception:
            logging.warning(f"Object set {object_set_name} not found. Skipping camera targeting.")

        # Step and RGB render
        step_sim(self.model, self.data, self.viewer)
        self.r.update_scene(self.data, self.camera_name)
        rgb_img = self.r.render()

        # Adjust camera yaw to look forward
        camera = self.r.scene.camera[1]
        forward = np.array(camera.forward, dtype=float)
        angle_to_turn = np.degrees(np.arctan2(forward[1], forward[0])) - np.degrees(self.data.qpos[2])
        if angle_to_turn < 0:
            angle_to_turn += 360
        self.data.qpos[3] = np.radians(angle_to_turn)
        self.data.ctrl[3] = np.radians(angle_to_turn)
        step_sim(self.model, self.data, self.viewer)

        # Re-render RGB after turning
        self.r.update_scene(self.data, self.camera_name)
        rgb_img = self.r.render()
        step_sim(self.model, self.data, self.viewer)

        # Depth render
        self.dr.update_scene(self.data, self.camera_name)
        depth_img = self.dr.render()
        depth_img[depth_img >= self.threshold] = 0

        # Compute camera matrices
        rotation = PointEstimation.compute_rotation_matrix(self.r)
        translation = PointEstimation.compute_translation_matrix(self.r)

        view = View(
            step=step_number,
            rgb=rgb_img,
            depth=depth_img,
            rotation=rotation,
            translation=translation
        )

        # Detect objects if model provided
        if self.object_detection_model:
            view.objects = detect_objects(view, self.object_detection_model)

        return view

    def capture_views(self, step_number: int, object_sets: list):
        """Capture multiple object sets in a single step, returning a dict of Views."""
        all_views = {}
        for object_set in object_sets:
            all_views[object_set] = self.capture_view(object_set, step_number)
        return all_views

    def process_view(self, view: View, object_set):
        """
        Process a single captured View to estimate 6D poses of detected objects.
        Returns observed_means, observed_classes, average_distance.
        """

        # Initialize processor
        processor = DenseProcessor(
            cam_intrinsic=[
                self.camera_width / 2,
                self.camera_height / 2,
                0.5 * self.camera_height / math.tan(self.model.vis.global_.fovy * math.pi / 360),
                0.5 * self.camera_height / math.tan(self.model.vis.global_.fovy * math.pi / 360),
            ],
            model_config=[10000, 21, self.camera_width, self.camera_height],
            rgb=view.rgb,
            depth=view.depth,
        )

        debug_images = []
        observed_means = []
        observed_cls = []
        distances = []

        for obj in view.objects:
            bbox = obj.bbox
            cls_id = obj.cls

            if not PointEstimation.is_point_in_3d_box(
                (round(obj.x), round(obj.y)),
                object_set,
                view,
                camera_intrinsic(self.model, self.camera_width, self.camera_height)
            ):  
                print("ERROR: Object detected was not part of the right object set")
                continue

            choose_mask = PointEstimation.region_growing(
                bbox,
                view,
                camera_intrinsic(self.model, self.camera_width, self.camera_height),
            )

            debug_images.append(np.where(choose_mask[:, :, np.newaxis] == 255, view.rgb, 0))

            rotation, coordinates = processor.process_data(
                bounded_box=bbox,
                id=cls_id,
                mask=choose_mask,
            )

            if all(coord == 0 for coord in coordinates):
                logging.error("Error detecting object location and/or depth")
                continue

            world_coords = np.dot(view.rotation, coordinates) + view.translation
            world_coords_no_trans = world_coords - view.translation
            distances.append(np.sqrt(world_coords_no_trans[0] ** 2 + world_coords_no_trans[1] ** 2))

            observed_cls.append(cls_id)
            angle = PointEstimation.calculateAngle(rotation, view.rotation)
            observed_means.append(np.array([world_coords[0], world_coords[1], angle]))

        average_distance = sum(distances) / len(distances) if distances else 0.0

        # Optional: display debug images
        # util.display_images_horizontally(debug_images)

        return observed_means, observed_cls, average_distance
