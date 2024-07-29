import logging
import random
from Constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEBUG_MODE,
    DURATION_IN_SECONDS,
    THRESHOLD,
    TOTAL_STEPS,
)
from DenseProcessor import DenseProcessor
from TrajectorySettings import TRAJECTORY_PATH
from filter_init import PhdFilter

# from ModelProcessor import ModelProcessor
import mujoco
import mujoco.viewer as viewer
import time
import itertools
import numpy as np
import math
import json
import logging
from scipy.interpolate import CubicSpline
from PIL import Image
from object_detection import detect_objects
import numpy as np
from scipy.spatial.transform import Rotation as R
import ssl
import PointEstimation
import Util

ssl._create_default_https_context = ssl._create_stdlib_context


def createGeom(scene: mujoco.MjvScene, location, rgba_given):
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03, 0.03, 0.03],
        pos=np.array([location[0], location[1], location[2]]),
        mat=np.eye(3).flatten(),
        rgba=rgba_given,
    )


def processView(
    singleView, model: mujoco.MjModel, scene: mujoco.MjvScene, r: mujoco.Renderer
):
    def through6dPose(singleView, model, scene, r):
        processor = DenseProcessor(
            cam_intrinsic=[
                CAMERA_WIDTH / 2,
                CAMERA_HEIGHT / 2,
                0.5
                * CAMERA_HEIGHT  ## maybe change to width
                / math.tan(model.vis.global_.fovy * math.pi / 360),
                0.5 * CAMERA_HEIGHT / math.tan(model.vis.global_.fovy * math.pi / 360),
            ],
            model_config=[10000, 21, CAMERA_HEIGHT, CAMERA_WIDTH],
            rgb=singleView["rgb_img"],
            depth=singleView["depth_img"],
        )

        debug_images = []
        measurements = []
        for object in singleView["objects"]:
            choose_mask = PointEstimation.region_growing(
                singleView["depth_img"],
                singleView["rgb_img"],
                singleView["camera_matrix"],
                object[3],
                model,
            )
            debug_images.append(
                np.where(choose_mask[:, :, np.newaxis] == 255, singleView["rgb_img"], 0)
            )

            camera_coordinates = processor.process_data(
                bounded_box=object[3],
                id=(object[4]),
                mask=choose_mask,
                scene=scene,
                model=model,
                singleView=singleView,
            )

            if all(coord == 0 for coord in camera_coordinates):
                logging.error("Error detecting object location and/or depth")
                continue

            world_coordinates = (
                np.dot(singleView["camera_matrix"]["rotation"], camera_coordinates[4:])
                + singleView["camera_matrix"]["translation"]
            )
            measurements.append(
                np.array(np.array([world_coordinates[0], world_coordinates[1]]))
            )
            PointEstimation.calculateAngle(
                camera_coordinates[:4], singleView["camera_matrix"]["rotation"]
            )
            if DEBUG_MODE:
                createGeom(
                    scene,
                    world_coordinates,
                    [random.random(), random.random(), random.random(), 1],
                )
        if DEBUG_MODE:
            Util.display_images_horizontally(debug_images)
        return measurements

    measurements = through6dPose(singleView, model, scene, r)

    # print(
    #     PointEstimation.is_point_range_visible(
    #         np.array([0, 2, 1.3]),
    #         singleView["depth_img"],
    #         PointEstimation.camera_intrinsic(model),
    #         singleView["camera_matrix"],
    #     )
    # )

    # print(
    #     PointEstimation.is_point_visible(
    #         np.array([0, 2, 0.3]),
    #         singleView["rgb_img"],
    #         singleView["depth_img"],
    #         camera_intrinsic(model),
    #         singleView["camera_matrix"],
    #     )
    # )
    # createGeom(
    #     scene,
    #     [0, 2, 0.3],
    #     [random.random(), random.random(), random.random(), 1],
    # )
    mujoco.mj_step(model, data)
    viewer.sync()
    return measurements


def step_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    r: mujoco.Renderer,
    dp: mujoco.Renderer,
    current_step: int,
):
    def numpy2pil(np_array: np.ndarray) -> Image:
        assert_msg = "Input shall be a HxWx3 ndarray"
        assert isinstance(np_array, np.ndarray), assert_msg
        assert len(np_array.shape) == 3, assert_msg
        assert np_array.shape[2] == 3, assert_msg

        img = Image.fromarray(np_array, "RGB")
        return img

    def compute_rotation_matrix(r):
        camera = r.scene.camera[1]
        # camera_right = np.cross(camera.up, camera.forward)
        # return np.column_stack((camera_right, camera.up, camera.forward))
        forward = np.array(camera.forward, dtype=float)
        up = np.array(camera.up, dtype=float)

        # Calculate right vector
        right = np.cross(-up, forward)

        # # Recalculate up vector to ensure orthogonality
        # up = np.cross(forward, right)

        # Construct 3x3 rotation matrix
        rotation_matrix = np.column_stack((right, -up, forward))

        return rotation_matrix

    def compute_translation_matrix(r):
        camera = r.scene.camera[1]
        return list(camera.pos)

    def calculate_control_signal(current_positions, desired_positions, kp=180.0):
        error = desired_positions - current_positions
        control_signals = kp * error

        return control_signals

    if current_step >= TOTAL_STEPS:
        logging.error("Cannot step: completed all steps")
        return []

    num_sync = int(100 * DURATION_IN_SECONDS / model.opt.timestep)
    update_interval = round(num_sync / TOTAL_STEPS)

    points = np.array(TRAJECTORY_PATH)
    points[:, 0] -= model.body("base_link").pos[0]
    points[:, 1] -= model.body("base_link").pos[1]
    spline = CubicSpline(np.arange(len(points)), points, axis=0)
    t = np.linspace(0, len(points) - 1, num_sync)

    if not viewer.is_running:
        return

    r.update_scene(data, "frontview")
    rgb_img = r.render()
    mujoco.mj_forward(model, data)
    if np.all(rgb_img == 0):
        logging.error("Image contains no objects")
        return
    dp.update_scene(data, "frontview")
    depth_img = dp.render()
    # depth_img[depth_img >= THRESHOLD] = 0
    if DEBUG_MODE:
        display_img = numpy2pil(rgb_img)
        Image.fromarray(depth_img, mode="L").save(
            f"debug_images/depth_{current_step}.png"
        )
        # Image.fromarray(depth_img, mode="L").show()
        Image.fromarray(rgb_img).save(f"debug_images/rgb_{current_step}.png")

    singleView = {
        "step": current_step,
        "depth_img": depth_img,
        "camera_matrix": {
            "rotation": compute_rotation_matrix(r),
            "translation": compute_translation_matrix(r),
        },
    }

    for i in range(update_interval):
        desired_position = spline(t[current_step * update_interval + i])
        current_positions = np.array([data.qpos[0], data.qpos[1]])
        control_signals = calculate_control_signal(current_positions, desired_position)
        data.ctrl[:2] = control_signals

        mujoco.mj_step(model, data)
        viewer.sync()

    return detect_objects(rgb_img, singleView)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Set up Mujoco model
    model = mujoco.MjModel.from_xml_path("google_robot/robot_scene.xml", dict())
    data = mujoco.MjData(model)
    dr = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    dr.enable_depth_rendering()
    dr.update_scene(data)

    r = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    r.update_scene(data)
    camera_collection = []
    paused = False

    target_birth_time = []
    targets_start = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        scene_option = mujoco.MjvOption()
        scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        mujoco.mj_step(model, data)
        viewer.sync()

        current_step = 0

        filter = PhdFilter()
        while viewer.is_running():
            x = input("Click s to step robot, click c to step through all: ")
            if x == "q" or x == "Q":
                break
            if x == "s":
                for i in range(1, 6):
                    model.geom(f"geo{i}").rgba = [1, 1, 1, 1]
                mujoco.mj_step(model, data)
                viewer.sync()
                singleView = step_robot(model, data, r, dr, current_step)
                if singleView != []:
                    current_step += 1
                    measurements = processView(singleView, model, viewer.user_scn, r)
                    filter.run_filter(
                        measurements,
                        singleView["depth_img"],
                        PointEstimation.camera_intrinsic(model),
                        singleView["camera_matrix"],
                    )
                    for i in range(1, 6):
                        model.geom(f"geo{i}").rgba = [1, 1, 1, 0.4]
                    mujoco.mj_step(model, data)
                    viewer.sync()
            if x == "c":
                for i in range(1, 6):
                    model.geom(f"geo{i}").rgba = [1, 1, 1, 1]
                mujoco.mj_step(model, data)
                viewer.sync()
                while current_step < TOTAL_STEPS:
                    singleView = step_robot(model, data, r, dr, current_step)
                    if singleView != []:
                        current_step += 1
                        measurements = processView(
                            singleView, model, viewer.user_scn, r
                        )
                        filter.run_filter(
                            measurements,
                            singleView["depth_img"],
                            PointEstimation.camera_intrinsic(model),
                            singleView["camera_matrix"],
                        )
                for i in range(1, 6):
                    model.geom(f"geo{i}").rgba = [1, 1, 1, 0.4]
                mujoco.mj_step(model, data)
                viewer.sync()
            if x == "p":
                results = filter.outputFilter()
                estimatedResult = results[len(results) - 1]
                print(estimatedResult)
                for result in estimatedResult:
                    createGeom(
                        viewer.user_scn,
                        [result[0], result[1], 1.5],
                        [random.random(), random.random(), random.random(), 1],
                    )
                mujoco.mj_step(model, data)
                viewer.sync()
