import logging
import random
from util_files.config_params import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEBUG_MODE,
    FRAME_DURATION,
)
from util_files.keys import (
    CAMERA_MATRIX_KEY,
    DEPTH_IMG_KEY,
    RGB_IMG_KEY,
    ROTATION_KEY,
    STEP_KEY,
    TRANSLATION_KEY,
)
from util_files.object_parameters import (
    CAMERA_NAME,
    OBJECT_SETS,
    TABLE_LOCATIONS,
    THRESHOLD,
)
from DenseProcessor import DenseProcessor
from filters.FilterProcessing import FilterProcessing
from util_files.TrajectorySettings import TRAJECTORY_PATH

import mujoco
import mujoco.viewer as viewer
import time
import numpy as np
import math
from logger_setup import logger
from scipy.interpolate import interp1d

from PIL import Image
from object_detection import detect_objects
import numpy as np
import ssl
import PointEstimation
import util_files.util as util
import torch
from util_files.transformation_utils import (
    euler_to_quaternion,
    quaternion_multiply,
)

ssl._create_default_https_context = ssl._create_stdlib_context


def update():
    mujoco.mj_step(model, data)
    viewer.sync()
    mujoco.mj_forward(model, data)


def camera_intrinsic(model):
    fov = model.vis.global_.fovy  # Fixed FOV angle
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT

    fW = 0.5 * height / math.tan(fov * math.pi / 360)
    fH = 0.5 * height / math.tan(fov * math.pi / 360)
    return np.array(((fW, 0, width / 2), (0, fH, height / 2), (0, 0, 1)))


def createGeom(location, rgba_given, new_size=[0.0095, 0.0095, 0.0095]):
    scn.ngeom += 1
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=new_size,
        pos=np.array([location[0], location[1], location[2]]),
        mat=np.eye(3).flatten(),
        rgba=rgba_given,
    )


def processView(
    singleView,
    object_set,
):
    def through6dPose(single_view, model):
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
            rgb=single_view[RGB_IMG_KEY],
            depth=single_view[DEPTH_IMG_KEY],
        )

        debug_images = []
        observed_means = []
        observed_cls = []
        distances = []
        for object in single_view["objects"]:
            if not PointEstimation.is_point_in_3d_box(
                (round(object[3][0]), round(object[3][1])),
                object_set,
                single_view[CAMERA_MATRIX_KEY],
                single_view[DEPTH_IMG_KEY],
                camera_intrinsic(model),
            ):
                print("ERROR: Object detected was not part of the right object set")
                continue
            choose_mask = PointEstimation.region_growing(
                single_view[DEPTH_IMG_KEY],
                single_view[RGB_IMG_KEY],
                single_view[CAMERA_MATRIX_KEY],
                object[3],
                camera_intrinsic(model),
            )
            debug_images.append(
                np.where(
                    choose_mask[:, :, np.newaxis] == 255, single_view[RGB_IMG_KEY], 0
                )
            )

            rotation, coordinates = processor.process_data(
                bounded_box=object[3],
                id=(object[4]),
                mask=choose_mask,
            )

            if all(coord == 0 for coord in coordinates):
                logging.error("Error detecting object location and/or depth")
                continue

            world_coordinates = (
                np.dot(single_view[CAMERA_MATRIX_KEY][ROTATION_KEY], coordinates)
                + single_view[CAMERA_MATRIX_KEY][TRANSLATION_KEY]
            )
            world_coordinates_without_translation = (
                world_coordinates - single_view[CAMERA_MATRIX_KEY][TRANSLATION_KEY]
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
                            single_view[CAMERA_MATRIX_KEY][ROTATION_KEY],
                        ),
                    ]
                )
            )
            if DEBUG_MODE:
                logger.info(world_coordinates)
                # createGeom(
                #     scn,
                #     world_coordinates,
                #     [random.random(), random.random(), random.random(), 1],
                # )
        distance = sum(distances) / len(distances)

        if DEBUG_MODE:
            util.display_images_horizontally(debug_images)
        return observed_means, observed_cls, distance

    return through6dPose(singleView, model)

    # createGeom(
    #     [0, 2, 0.3],
    #     [random.random(), random.random(), random.random(), 1],
    # )


def turn_to_angle(target_angle_degrees, data):
    data.qpos[2] = target_angle_degrees
    data.ctrl[2] = target_angle_degrees


def turn_robot(target_angle_degrees, data, duration=1.5, dt=0.01):
    current_angle = np.degrees(data.qpos[2])
    steps = int(duration / dt)
    step_size = (target_angle_degrees - current_angle) / steps

    for _ in range(steps):
        current_angle += step_size
        data.qpos[2] = np.radians(current_angle)
        data.ctrl[2] = np.radians(current_angle)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)


def turn_camera(target_angle_degrees, data, duration=0.05, dt=0.01):
    current_angle = np.degrees(data.qpos[3])
    steps = int(duration / dt)
    step_size = (target_angle_degrees - current_angle) / steps

    for _ in range(steps):
        current_angle += step_size
        data.qpos[3] = np.radians(current_angle)
        data.ctrl[3] = np.radians(current_angle)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)


def next_viewpoint(
    viewpoint_number: int,
    objectDetectionModel,
):
    if viewpoint_number >= len(TRAJECTORY_PATH):
        logging.error("Cannot step: completed all steps")
        return []

    if (
        viewpoint_number > 0
        and TRAJECTORY_PATH[viewpoint_number][1]
        != TRAJECTORY_PATH[viewpoint_number - 1][1]
    ):
        turn_robot(TRAJECTORY_PATH[viewpoint_number][1], data)

    points = np.array([element[0] for element in TRAJECTORY_PATH])

    points[:, 0] -= model.body("base_link").pos[0]
    points[:, 1] -= model.body("base_link").pos[1]

    spline = interp1d(np.arange(len(points)), points, kind="linear", axis=0)

    if viewpoint_number >= 0:
        current_position = np.array([data.qpos[0], data.qpos[1]])
        target_position = spline(viewpoint_number)

        distance = np.linalg.norm(target_position - current_position)
        duration = distance / FRAME_DURATION
        steps = int(duration / 0.01)
        if steps != 0:
            step_size = (target_position - current_position) / steps

            for _ in range(steps):
                current_position += step_size
                data.qpos[0:2] = current_position
                data.ctrl[0:2] = current_position
                mujoco.mj_step(model, data)  # Step the simulation forward
                viewer.sync()
                time.sleep(0.01)  # Wait for the next time step

    mujoco.mj_forward(model, data)
    all_results = {}
    for object_set in TRAJECTORY_PATH[viewpoint_number][2]:
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
        dr.update_scene(data, CAMERA_NAME)
        depth_img = dr.render()
        depth_img[depth_img >= THRESHOLD] = 0
        # if DEBUG_MODE:
        # display_img = numpy2pil(rgb_img)
        # Image.fromarray(depth_img, mode="L").save(
        #     f"debug_images/depth_{current_step}.png"
        # )
        # Image.fromarray(depth_img, mode="L").show()
        # Image.fromarray(rgb_img).save(f"debug_images/rgb_{current_step}.png")

        single_view = {
            STEP_KEY: viewpoint_number,
            DEPTH_IMG_KEY: depth_img,
            CAMERA_MATRIX_KEY: {
                ROTATION_KEY: PointEstimation.compute_rotation_matrix(r),
                TRANSLATION_KEY: PointEstimation.compute_translation_matrix(r),
            },
        }

        all_results[object_set] = detect_objects(
            rgb_img, single_view, objectDetectionModel
        )
    return all_results


def setup_objects(table_name, object_set):
    ground_truth_objs = []
    ground_truth_types = []
    for geom in object_set:
        new_quat = euler_to_quaternion(0, 0, geom[3])
        originalZ = model.geom(geom[0]).pos[2]
        model.geom(geom[0]).quat = quaternion_multiply(
            new_quat, model.geom(geom[0]).quat
        )
        model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]
        ground_truth_types.append(geom[1])
        delta_locs = np.array(
            [
                model.geom(geom[0]).pos[0],
                model.geom(geom[0]).pos[1],
            ]
        ) + np.array(TABLE_LOCATIONS[table_name])

        ground_truth_objs.append([delta_locs[0], delta_locs[1], geom[3]])

    update()
    combined = list(zip(list(ground_truth_types), list(ground_truth_objs)))
    combined.sort(key=lambda x: x[0])
    return zip(*combined)


# def progress_geoms(visible):
#     for object_set in OBJECT_SETS.values():
#         for geom in object_set:
#             if visible:
#                 model.geom(geom[0]).rgba = [1, 1, 1, 1]
#             else:
#                 model.geom(geom[0]).rgba = [1, 1, 1, 0.4]
#     update()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    global model, data, r, dr, scn

    model = mujoco.MjModel.from_xml_path("environment_assets/EXP2_scene.xml", dict())
    data = mujoco.MjData(model)

    dr = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    dr.enable_depth_rendering()
    dr.update_scene(data)

    r = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    r.update_scene(data)

    target_birth_time = []
    targets_start = []

    objectDetectionModel = torch.hub.load(
        "ultralytics/yolov5", "custom", path="yolo-for-ycb/best.pt"
    )

    filters = {}
    with mujoco.viewer.launch_passive(model, data) as viewer:
        scn = viewer.user_scn
        for object_name, object_set in OBJECT_SETS.items():
            ground_truth_types, ground_truth_objs = setup_objects(
                object_name, object_set
            )
            filters[object_name] = FilterProcessing(
                ground_truth_objs, ground_truth_types, object_name
            )

        scene_option = mujoco.MjvOption()
        scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        update()

        viewpoint_number = 0
        data.ctrl[5] = 0.5
        data.ctrl[7] = 3
        model.vis.global_.fovy = model.cam(CAMERA_NAME).fovy[0]
        while viewer.is_running():
            x = input(
                "Commands:\n"
                "  s - Step robot one step\n"
                "  a - Continue stepping through all steps\n"
                "  p - Print filter results\n"
                "  e - Evaluate filters\n"
                "  q - Quit the program\n"
                "Enter your choice: "
            )
            if x in {"quit", "q", "Q"}:
                break
            elif x in {"step", "s", "S"}:
                views = next_viewpoint(viewpoint_number, objectDetectionModel)
                viewpoint_number += 1
                if views != []:
                    for view_name, view in views.items():
                        observed_means, observed_cls, distance = processView(
                            view,
                            view_name,
                        )
                        filter = filters[view_name]
                        filter.run_filter(
                            data.qpos.copy(),
                            data.ctrl.copy(),
                            observed_means,
                            observed_cls,
                            distance,
                        )
            elif x in {"step_all", "a", "A"}:
                while viewpoint_number < len(TRAJECTORY_PATH):
                    views = next_viewpoint(viewpoint_number, objectDetectionModel)
                    viewpoint_number += 1
                    if views != []:
                        for view_name, view in views.items():
                            observed_means, observed_cls, distance = processView(
                                view, view_name
                            )
                            filter = filters[view_name]
                            filter.run_filter(
                                data.qpos.copy(),
                                data.ctrl.copy(),
                                observed_means,
                                observed_cls,
                                distance,
                            )
            elif x == "p":
                for filter in filters.values():
                    means, clss = filter.outputFilter()
                    for result in means:
                        logger.info(result)
                    update()
            elif x == "e":
                for filter in filters.values():
                    filter.evaluate()
