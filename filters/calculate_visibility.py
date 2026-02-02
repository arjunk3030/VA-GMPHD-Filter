import numpy as np
import mujoco
from PIL import Image
from mujoco import mjtGeom
from scipy.spatial.transform import Rotation as R
from util_files.logger_setup import logger
from util_files.object_parameters import (
    CAMERA_NAME,
    CLS_TO_MATERIAL,
    CLS_TO_MESH,
    FLOOR_HEIGHT,
    MUJOCO_TO_POSE,
    OBJECT_SETS,
    THRESHOLD,
)
from util_files.transformation_utils import (
    asymmetric_to_symmetric_rotation,
    euler_to_quaternion,
    quaternion_multiply,
)


def calculate_visibility_probability(visible_points, occluded_depth_image):
    if not visible_points:
        return 0

    valid_points = 0
    visible_points_count = 0

    for x, y, depth_value in visible_points:
        valid_points += 1
        if abs(occluded_depth_image[y, x] - depth_value) <= 0.01:
            visible_points_count += 1

    if valid_points < 600:
        return 0

    visibility_probability = visible_points_count / valid_points
    return visibility_probability


def get_visible_points(simple_depth_img):
    non_zero_indices = np.nonzero(simple_depth_img)
    total_non_zero_points = len(non_zero_indices[0])

    if total_non_zero_points == 0:
        return []

    visible_points = []
    for i in range(total_non_zero_points):
        y, x = non_zero_indices[0][i], non_zero_indices[1][i]
        depth_value = simple_depth_img[y, x]
        visible_points.append((x, y, depth_value))

    return visible_points


def set_values(geom, mean, cls):
    geom.type = mjtGeom.mjGEOM_MESH
    geom.meshname = CLS_TO_MESH(cls)
    geom.material = CLS_TO_MATERIAL(cls)
    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees
    geom.quat = euler_to_quaternion(0, 0, mean[2])
    new_pos = asymmetric_to_symmetric_rotation(
        true_center,
        MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    # geom.pos = true_center
    geom.pos = new_pos


def calculate_all_p_v(
    current_object_set, scene_pos, scene_ctrl, all_means, all_cls, estimated_mean, camera_width, camera_height
):
    allGaussiansModelSpec = mujoco.MjSpec()
    allGaussiansModelSpec.from_file("environment_assets/empty_EXP2_scene.xml")

    allGaussians = allGaussiansModelSpec.worldbody.add_body()
    allGaussians.pos = [0, 0, FLOOR_HEIGHT]

    visible_points_list = []
    for i, (mean, cls) in enumerate(zip(all_means, all_cls)):
        # Generate views with a single Gaussian
        singleGaussian = mujoco.MjSpec()
        singleGaussian.from_file("environment_assets/empty_EXP2_scene.xml")

        object_body = singleGaussian.worldbody.add_body()
        object_body.pos = [0, 0, FLOOR_HEIGHT]
        geom = object_body.add_geom()
        set_values(geom, mean, cls.index(max(cls)))

        model = singleGaussian.compile()
        data = mujoco.MjData(model)
        model.camera(CAMERA_NAME).targetbodyid = model.body(current_object_set).id
        data.ctrl[5] = 0.5
        data.ctrl[7] = 3
        model.vis.global_.fovy = model.cam(CAMERA_NAME).fovy[0]
        data.ctrl[:] = scene_ctrl
        data.qpos[:] = scene_pos
        mujoco.mj_step(model, data)

        for object_set in OBJECT_SETS.values():
            for geom in object_set:
                newQuat = euler_to_quaternion(0, 0, geom[3])
                originalZ = model.geom(geom[0]).pos[2]
                model.geom(geom[0]).quat = quaternion_multiply(
                    newQuat, model.geom(geom[0]).quat
                )
                model.geom(geom[0]).rgba = [1, 1, 1, 0]
                model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]

        mujoco.mj_step(model, data)
        dr = mujoco.Renderer(model, camera_height, camera_width)
        dr.enable_depth_rendering()
        dr.update_scene(data, CAMERA_NAME)

        simple_depth_img = dr.render()
        simple_depth_img[simple_depth_img >= THRESHOLD] = 0
        mujoco.mj_step(model, data)

        visible_points_list.append(get_visible_points(simple_depth_img))

        # r = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
        # r.update_scene(data, CAMERA_NAME)
        # Image.fromarray(r.render()).show()

        # Add to model for 2nd list
        if any(np.array_equal(mean, sublist) for sublist in estimated_mean):
            geom = allGaussians.add_geom()
            set_values(geom, mean, cls.index(max(cls)))

    model = allGaussiansModelSpec.compile()
    data = mujoco.MjData(model)
    data.ctrl[5] = 0.5
    data.ctrl[7] = 3
    model.vis.global_.fovy = model.cam(CAMERA_NAME).fovy[0]
    model.camera(CAMERA_NAME).targetbodyid = model.body(current_object_set).id
    data.ctrl[:] = scene_ctrl
    data.qpos[:] = scene_pos
    mujoco.mj_step(model, data)

    for object_set in OBJECT_SETS.values():
        for geom in object_set:
            newQuat = euler_to_quaternion(0, 0, geom[3])
            originalZ = model.geom(geom[0]).pos[2]
            model.geom(geom[0]).quat = quaternion_multiply(
                newQuat, model.geom(geom[0]).quat
            )
            model.geom(geom[0]).rgba = [1, 1, 1, 0]
            model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]

    data.qpos[:] = scene_pos
    mujoco.mj_step(model, data)
    dr = mujoco.Renderer(model, camera_height, camera_width)
    dr.enable_depth_rendering()
    dr.update_scene(data, CAMERA_NAME)
    occluded_depth_img = dr.render()
    occluded_depth_img[occluded_depth_img >= THRESHOLD] = 0
    mujoco.mj_step(model, data)

    # r = mujoco.Renderer(model, camera_height, camera_width)
    # r.update_scene(data, CAMERA_NAME)
    # Image.fromarray(r.render()).show()

    visibilities = []
    for visible_points in visible_points_list:
        visibilities.append(
            calculate_visibility_probability(visible_points, occluded_depth_img)
        )
    logger.info(
        f"visibilities are {visibilities} {[cls.index(max(cls)) for cls in all_cls]}"
    )
    return visibilities
