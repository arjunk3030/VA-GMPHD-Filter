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
    def camera_intrinsic(model):
        fov = model.vis.global_.fovy  # Field of view angle
        width = CAMERA_WIDTH
        height = CAMERA_HEIGHT

        fW = (
            0.5 * height / math.tan(fov * math.pi / 360)
        )  # TODO if both should be height
        fH = 0.5 * height / math.tan(fov * math.pi / 360)
        return np.array(((fW, 0, width / 2), (0, fH, height / 2), (0, 0, 1)))

    def image_to_camera_coordinates(x, y, depth, K):
        x_norm = (x - K[0][2]) / K[0][0]
        y_norm = (y - K[1][2]) / K[1][1]
        x_c = x_norm * depth
        y_c = y_norm * depth
        z_c = depth
        return x_c, y_c, z_c

    def throughYolo(singleView, model, scene, r):
        for object in singleView["objects"]:
            camera_coordinates = np.array(
                image_to_camera_coordinates(
                    object[0], object[1], object[2], camera_intrinsic(model)
                )
            )
            if all(coord == 0 for coord in camera_coordinates):
                logging.error("Error detecting object location and/or depth")
                continue

            world_coordinates = (
                np.dot(singleView["camera_matrix"]["rotation"], camera_coordinates)
                + singleView["camera_matrix"]["translation"]
            )

            if DEBUG_MODE:
                createGeom(
                    scene,
                    world_coordinates,
                    [random.random(), random.random(), random.random(), 0.3],
                )

    def through6dPose(singleView, model, scene, r):
        def quaternion_to_rotation_matrix(q):
            w, x, y, z = q
            return np.array(
                [
                    [
                        1 - 2 * y**2 - 2 * z**2,
                        2 * x * y - 2 * w * z,
                        2 * x * z + 2 * w * y,
                    ],
                    [
                        2 * x * y + 2 * w * z,
                        1 - 2 * x**2 - 2 * z**2,
                        2 * y * z - 2 * w * x,
                    ],
                    [
                        2 * x * z - 2 * w * y,
                        2 * y * z + 2 * w * x,
                        1 - 2 * x**2 - 2 * y**2,
                    ],
                ]
            )

        def rotation_matrix_to_euler(R):
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0

            return np.array([x, y, z])

        def calculateAngle(q, rotation):
            # rotation_camera_matrix = R.from_matrix(rotation)

            # rotation_camera_matrix_inv = rotation_camera_matrix.inv()

            # rotation_world = rotation_camera_matrix_inv * rotation_camera

            # euler_world = rotation_world.as_quat()

            # print("World-view-based quat angles:", euler_world)
            # # Get the quaternion for the world-view-based rotation
            # q_world = rotation_world.as_quat()
            new_rot = np.dot(rotation, quaternion_to_rotation_matrix(q))
            new_quat = R.from_matrix(new_rot).as_quat()
            new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
            print("World-view-based quaternion:", new_quat)
            print("Euler degrees: ", np.degrees(rotation_matrix_to_euler(new_rot)))

        def distance_3d(point1, point2, extrinsics):
            rotation_matrix = extrinsics["camera_matrix"]["rotation"]
            translation = extrinsics["camera_matrix"]["translation"]

            point1_world = rotation_matrix @ point1 + translation
            point2_world = rotation_matrix @ point2 + translation

            return np.linalg.norm(point1_world - point2_world)

        def createChooseMask(
            singleView, seed, model, tolerance_depth=0.05, tolerance_color=100
        ):
            depth_image = singleView["depth_img"]
            rgb_image = np.array(singleView["rgb_img"])

            height, width = depth_image.shape
            segmented = np.zeros((height, width), dtype=np.uint8)
            stack = [seed]
            segmented[seed[1], seed[0]] = 255

            while stack:
                x, y = stack.pop()
                current_3d = image_to_camera_coordinates(
                    x,
                    y,
                    singleView["depth_img"][round(y), round(x)],
                    camera_intrinsic(model),
                )
                current_color = rgb_image[y, x]

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and segmented[ny, nx] == 0:
                        neighbor_3d = image_to_camera_coordinates(
                            nx,
                            ny,
                            singleView["depth_img"][round(ny), round(nx)],
                            camera_intrinsic(model),
                        )
                        neighbor_color = rgb_image[ny, nx]

                        depth_diff = distance_3d(current_3d, neighbor_3d, singleView)
                        color_diff = np.linalg.norm(
                            current_color.astype(float) - neighbor_color.astype(float)
                        )

                        if (
                            depth_diff <= tolerance_depth
                            and color_diff <= tolerance_color
                        ):
                            stack.append((nx, ny))
                            segmented[ny, nx] = (
                                255  # Mark the pixel as segmented immediately
                            )

            return segmented

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

        for object in singleView["objects"]:
            choose_mask = createChooseMask(
                singleView, (round(object[0]), round(object[1])), model
            )
            result = np.where(
                choose_mask[:, :, np.newaxis] == 255, singleView["rgb_img"], 0
            )
            result_image = Image.fromarray(result.astype("uint8"))
            Image.fromarray(choose_mask).show(title="Segmented Depth")
            result_image.show(title="Segmented Result")

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
            calculateAngle(
                camera_coordinates[:4], singleView["camera_matrix"]["rotation"]
            )
            if DEBUG_MODE:
                createGeom(
                    scene,
                    world_coordinates,
                    [random.random(), random.random(), random.random(), 1],
                )

    through6dPose(singleView, model, scene, r)
    mujoco.mj_step(model, data)
    viewer.sync()


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
    depth_img[depth_img >= THRESHOLD] = 0
    if DEBUG_MODE:
        display_img = numpy2pil(rgb_img)
        display_img.show()
        Image.fromarray(depth_img, mode="L").save(
            f"debug_images/depth_{current_step}.png"
        )
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
        while viewer.is_running():
            x = input("Click s to step robot, click f to step through all: ")
            if x == "q" or x == "Q":
                break
            if x == "s":
                singleView = step_robot(model, data, r, dr, current_step)
                if singleView != []:
                    current_step += 1
                    processView(singleView, model, viewer.user_scn, r)
            if x == "c":
                while current_step >= TOTAL_STEPS:
                    singleView = step_robot(model, data, r, dr, current_step)
                    if singleView != []:
                        current_step += 1
                        processView(singleView, model, viewer.user_scn, r)
