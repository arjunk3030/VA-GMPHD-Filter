import logging
import random
from Constants import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    DEBUG_MODE,
    MUJOCO_TO_POSE,
    OBJECT_SETS,
    SPEED,
    TABLE_DELTAS,
    THRESHOLD,
)
from DenseProcessor import DenseProcessor
from FilterProcessing import FilterProcessing
from TrajectorySettings import TRAJECTORY_PATH

import mujoco
import mujoco.viewer as viewer
import time
import numpy as np
import math
import logging
from scipy.interpolate import interp1d

from PIL import Image
from object_detection import detect_objects
import numpy as np
import ssl
import PointEstimation
import Util
import PHDFilterCalculations
import torch

ssl._create_default_https_context = ssl._create_stdlib_context


def createGeom(
    scene: mujoco.MjvScene, location, rgba_given, new_size=[0.0095, 0.0095, 0.0095]
):
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=new_size,
        pos=np.array([location[0], location[1], location[2]]),
        mat=np.eye(3).flatten(),
        rgba=rgba_given,
    )


def processView(
    singleView,
    object_set,
    model: mujoco.MjModel,
    scene: mujoco.MjvScene,
    r: mujoco.Renderer,
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
            model_config=[10000, 21, CAMERA_WIDTH, CAMERA_HEIGHT],
            rgb=singleView["rgb_img"],
            depth=singleView["depth_img"],
        )

        debug_images = []
        observed_means = []
        observed_cls = []
        distances = []
        for object in singleView["objects"]:
            if not PointEstimation.is_point_in_3d_box(
                (round(object[3][0]), round(object[3][1])),
                object_set,
                singleView["camera_matrix"],
                singleView["depth_img"],
                model,
            ):
                print("ERROR: Object detected was not part of the right object set")
                continue
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

            rotation, coordinates = processor.process_data(
                bounded_box=object[3],
                id=(object[4]),
                mask=choose_mask,
                scene=scene,
                model=model,
                singleView=singleView,
            )

            if all(coord == 0 for coord in coordinates):
                logging.error("Error detecting object location and/or depth")
                continue

            world_coordinates = (
                np.dot(singleView["camera_matrix"]["rotation"], coordinates)
                + singleView["camera_matrix"]["translation"]
            )
            world_coordinates_without_translation = (
                world_coordinates - singleView["camera_matrix"]["translation"]
            )
            distances.append(
                np.sqrt(
                    world_coordinates_without_translation[0] ** 2
                    + world_coordinates_without_translation[1] ** 2
                )
            )

            observed_cls.append(object[4])
            observed_means.append(
                np.array(
                    [
                        world_coordinates[0],
                        world_coordinates[1],
                        PointEstimation.calculateAngle(
                            rotation,
                            singleView["camera_matrix"]["rotation"],
                        ),
                    ]
                )
            )
            if DEBUG_MODE:
                print(world_coordinates)
                createGeom(
                    scene,
                    world_coordinates,
                    [random.random(), random.random(), random.random(), 1],
                )
        distance = sum(distances) / len(distances)

        if DEBUG_MODE:
            Util.display_images_horizontally(debug_images)
        return observed_means, observed_cls, distance

    return through6dPose(singleView, model, scene, r)

    # createGeom(
    #     scene,
    #     [0, 2, 0.3],
    #     [random.random(), random.random(), random.random(), 1],
    # )


def turn_to_angle(target_angle_degrees, data):
    data.qpos[2] = target_angle_degrees
    data.ctrl[2] = target_angle_degrees


def turn_robot(target_angle_degrees, data, duration=1.5, dt=0.01):
    current_angle = np.degrees(
        data.qpos[2]
    )  # Convert current angle from radians to degrees
    steps = int(duration / dt)  # Number of steps to interpolate over
    step_size = (target_angle_degrees - current_angle) / steps  # Increment per step

    for _ in range(steps):
        current_angle += step_size
        data.qpos[2] = np.radians(current_angle)  # Update qpos[2] (in radians)
        data.ctrl[2] = np.radians(current_angle)  # Apply the control signal
        mujoco.mj_step(model, data)  # Step the simulation forward
        viewer.sync()
        time.sleep(dt)  # Wait for the next time step


def turn_camera(target_angle_degrees, data, duration=0.05, dt=0.01):
    current_angle = np.degrees(
        data.qpos[3]
    )  # Convert current angle from radians to degrees
    steps = int(duration / dt)  # Number of steps to interpolate over
    step_size = (target_angle_degrees - current_angle) / steps  # Increment per step

    for _ in range(steps):
        current_angle += step_size
        data.qpos[3] = np.radians(current_angle)  # Update qpos[2] (in radians)
        data.ctrl[3] = np.radians(current_angle)  # Apply the control signal
        mujoco.mj_step(model, data)  # Step the simulation forward
        viewer.sync()
        time.sleep(dt)  # Wait for the next time step


def step_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    r: mujoco.Renderer,
    dp: mujoco.Renderer,
    current_step: int,
    objectDetectionModel,
):
    def numpy2pil(np_array: np.ndarray) -> Image:
        assert_msg = "Input shall be a HxWx3 ndarray"
        assert isinstance(np_array, np.ndarray), assert_msg
        assert len(np_array.shape) == 3, assert_msg
        assert np_array.shape[2] == 3, assert_msg

        img = Image.fromarray(np_array, "RGB")
        return img

    if current_step >= len(TRAJECTORY_PATH):
        logging.error("Cannot step: completed all steps")
        return []

    if (
        current_step > 0
        and TRAJECTORY_PATH[current_step][1] != TRAJECTORY_PATH[current_step - 1][1]
    ):
        turn_robot(TRAJECTORY_PATH[current_step][1], data)

    points = np.array([element[0] for element in TRAJECTORY_PATH])

    points[:, 0] -= model.body("base_link").pos[0]
    points[:, 1] -= model.body("base_link").pos[1]

    spline = interp1d(np.arange(len(points)), points, kind="linear", axis=0)

    if current_step >= 0:
        current_position = np.array([data.qpos[0], data.qpos[1]])
        target_position = spline(current_step)

        distance = np.linalg.norm(target_position - current_position)
        duration = distance / SPEED
        steps = int(duration / 0.01)
        if steps != 0:
            step_size = (target_position - current_position) / steps

            for _ in range(steps):
                current_position += step_size
                data.qpos[0:2] = current_position
                data.ctrl[0:2] = current_position
                mujoco.mj_step(model, data)  # Step the simulation forward
                viewer.sync()  # Sync the viewer
                time.sleep(0.01)  # Wait for the next time step

    mujoco.mj_forward(model, data)
    all_results = {}
    for object_set in TRAJECTORY_PATH[current_step][2]:
        model.camera(CAMERA_NAME).targetbodyid = model.body(object_set).id
        mujoco.mj_step(model, data)  # Step the simulation forward
        viewer.sync()

        r.update_scene(data, CAMERA_NAME)
        rgb_img = r.render()

        mujoco.mj_step(model, data)  # Step the simulation forward
        viewer.sync()
        camera = r.scene.camera[1]
        forward = np.array(camera.forward, dtype=float)

        angle_to_turn = np.degrees(np.arctan2(forward[1], forward[0])) - np.degrees(
            data.qpos[2]
        )
        if angle_to_turn < 0:
            angle_to_turn = 360 + angle_to_turn
        turn_camera(angle_to_turn, data)
        time.sleep(0.5)

        r.update_scene(data, CAMERA_NAME)
        rgb_img = r.render()

        mujoco.mj_forward(model, data)
        if np.all(rgb_img == 0):
            logging.error("Image contains no objects")
            return
        dp.update_scene(data, CAMERA_NAME)
        depth_img = dp.render()
        depth_img[depth_img >= THRESHOLD] = 0
        if DEBUG_MODE:
            display_img = numpy2pil(rgb_img)
            # Image.fromarray(depth_img, mode="L").save(
            #     f"debug_images/depth_{current_step}.png"
            # )
            # Image.fromarray(depth_img, mode="L").show()
            # Image.fromarray(rgb_img).save(f"debug_images/rgb_{current_step}.png")

        singleView = {
            "step": current_step,
            "depth_img": depth_img,
            "camera_matrix": {
                "rotation": PointEstimation.compute_rotation_matrix(r),
                "translation": PointEstimation.compute_translation_matrix(r),
            },
        }

        all_results[object_set] = detect_objects(
            rgb_img, singleView, objectDetectionModel
        )
    return all_results


def stateUpdates(model, data, table_name, object_set):
    ground_truth_objs = []
    ground_truth_types = []
    for geom in object_set:
        newQuat = PointEstimation.euler_to_quaternion(0, 0, geom[3])
        originalZ = model.geom(geom[0]).pos[2]
        model.geom(geom[0]).quat = PHDFilterCalculations.quaternion_multiply(
            newQuat, model.geom(geom[0]).quat
        )
        model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]
        ground_truth_types.append(geom[1])
        delta_locs = np.array(
            [
                model.geom(geom[0]).pos[0],
                model.geom(geom[0]).pos[1],
            ]
        ) + np.array(TABLE_DELTAS[table_name])

        ground_truth_objs.append([delta_locs[0], delta_locs[1], geom[3]])

        # createGeom(
        #     viewer.user_scn,
        #     [
        #         new_pos[0],
        #         new_pos[1],
        #         1.3,
        #     ],
        #     [1, 1, 1, 1],
        # )

    mujoco.mj_step(model, data)
    viewer.sync()
    combined = list(zip(list(ground_truth_types), list(ground_truth_objs)))
    combined.sort(key=lambda x: x[0])
    return zip(*combined)


def progress_geoms(visible):
    for object_set in OBJECT_SETS.values():
        for geom in object_set:
            if visible:
                model.geom(geom[0]).rgba = [1, 1, 1, 1]
            else:
                model.geom(geom[0]).rgba = [1, 1, 1, 0.4]
    mujoco.mj_step(model, data)
    viewer.sync()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Set up Mujoco model
    model = mujoco.MjModel.from_xml_path("google_robot/EXP2_scene.xml", dict())
    data = mujoco.MjData(model)

    # state updates

    dr = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    dr.enable_depth_rendering()
    dr.update_scene(data)

    r = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    r.update_scene(data)
    camera_collection = []
    paused = False

    target_birth_time = []
    targets_start = []

    objectDetectionModel = torch.hub.load(
        "ultralytics/yolov5", "custom", path="yolo-for-ycb/best.pt"
    )

    filters = {}
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for object_name, object_set in OBJECT_SETS.items():
            ground_truth_types, ground_truth_objs = stateUpdates(
                model, data, object_name, object_set
            )
            filters[object_name] = FilterProcessing(
                ground_truth_objs, ground_truth_types, object_name
            )

        scene_option = mujoco.MjvOption()
        scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # height = 1.25
        # spacing = 0.1
        # range_min = -1
        # range_max = 1

        # # Create dots in the grid
        # for x in range(int((range_max - range_min) / spacing) + 1):
        #     for y in range(int((range_max - range_min) / spacing) + 1):
        #         pos_x = range_min + x * spacing
        #         pos_y = range_min + y * spacing
        #         pos = [pos_x, pos_y, height]

        #         # Adjust size for the dot at (0, 0)
        #         if pos_x == 0 or pos_y == 0:
        #             color = [1, 1, 1, 1]  # Larger size for the middle dot
        #             createGeom(viewer.user_scn, pos, color, [0.03, 0.03, 0.03])
        #         else:
        #             color = [
        #                 random.random(),
        #                 random.random(),
        #                 random.random(),
        #                 1,
        #             ]  # Random color

        #         createGeom(viewer.user_scn, pos, color)

        mujoco.mj_step(model, data)
        viewer.sync()
        mujoco.mj_forward(model, data)

        current_step = 0
        data.ctrl[5] = 0.5
        data.ctrl[7] = 3
        model.vis.global_.fovy = model.cam(CAMERA_NAME).fovy[0]

        while viewer.is_running():
            x = input("Click s to step robot, click c to step through all: ")
            if x == "q" or x == "Q":
                break
            if x == "s":
                progress_geoms(visible=True)
                views = step_robot(
                    model, data, r, dr, current_step, objectDetectionModel
                )
                current_step += 1
                if views != []:
                    for view_name, view in views.items():
                        observed_means, observed_cls, distance = processView(
                            view, view_name, model, viewer.user_scn, r
                        )
                        filter = filters[view_name]
                        filter.run_filter(
                            data.qpos.copy(),
                            data.ctrl.copy(),
                            observed_means,
                            observed_cls,
                            distance,
                        )
                progress_geoms(visible=False)

            if x == "c":
                progress_geoms(visible=True)
                while current_step < len(TRAJECTORY_PATH):
                    views = step_robot(
                        model, data, r, dr, current_step, objectDetectionModel
                    )
                    current_step += 1
                    if views != []:
                        for view_name, view in views.items():
                            observed_means, observed_cls, distance = processView(
                                view, view_name, model, viewer.user_scn, r
                            )
                            filter = filters[view_name]
                            filter.run_filter(
                                data.qpos.copy(),
                                data.ctrl.copy(),
                                observed_means,
                                observed_cls,
                                distance,
                            )
                progress_geoms(visible=False)
            if x == "p":
                for filter in filters.values():
                    means, clss = filter.outputFilter()
                    for result in means:
                        print(result)
                        createGeom(
                            viewer.user_scn,
                            [result[0], result[1], 1.2],
                            [random.random(), random.random(), random.random(), 1],
                        )
                    mujoco.mj_step(model, data)
                    viewer.sync()
            if x == "e":
                for filter in filters.values():
                    filter.evaluate()
