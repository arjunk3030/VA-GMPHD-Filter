import PointEstimation
import numpy as np
import Constants
import mujoco
from PIL import Image
from mujoco import mjtGeom
from scipy.spatial.transform import Rotation as R


def calculate_visibility_probability(visible_points, occluded_depth_image):
    if not visible_points:
        return 0

    valid_points = 0
    visible_points_count = 0

    for x, y, depth_value in visible_points:
        valid_points += 1
        if abs(occluded_depth_image[y, x] - depth_value) <= 0.01:
            visible_points_count += 1

    if valid_points < 500:
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


def asymmetric_to_symmetric_rotation(
    true_center,
    offset,
    euler_angles,
):
    # Convert Euler angles to rotation matrix
    def euler_to_rotation_matrix(angles):
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )

        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )

        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )

        return np.dot(Rz, np.dot(Ry, Rx))

    asymmetric_center = true_center + offset
    # Convert Euler angles to radians
    euler_angles_rad = np.radians(euler_angles)

    # Create rotation matrix
    rotation_matrix = euler_to_rotation_matrix(euler_angles_rad)

    # Calculate the translation
    rotated_offset = np.dot(rotation_matrix, offset)
    translation = rotated_offset - offset

    return translation + asymmetric_center


def set_values(geom, mean, cls):
    geom.type = mjtGeom.mjGEOM_MESH
    geom.meshname = Constants.CLS_TO_MESH[cls]
    geom.material = Constants.CLS_TO_MATERIAL[cls]
    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees

    new_pos = asymmetric_to_symmetric_rotation(
        true_center,
        Constants.MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    geom.pos = new_pos
    geom.quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])


def calculate_all_p_v(scene_pos, all_means, all_cls, estimated_mean):
    allGaussiansModelSpec = mujoco.MjSpec()
    allGaussiansModelSpec.from_file("google_robot/empty_EXP2_scene.xml")

    allGaussians = allGaussiansModelSpec.worldbody.add_body()
    allGaussians.pos = [0, 0, Constants.FLOOR_HEIGHT]

    visible_points_list = []
    for i, (mean, cls) in enumerate(zip(all_means, all_cls)):
        # Generate views with a single Gaussian
        singleGaussian = mujoco.MjSpec()
        singleGaussian.from_file("google_robot/empty_EXP2_scene.xml")

        object_body = singleGaussian.worldbody.add_body()
        object_body.pos = [0, 0, Constants.FLOOR_HEIGHT]
        geom = object_body.add_geom()
        set_values(geom, mean, cls.index(max(cls)))

        model = singleGaussian.compile()
        data = mujoco.MjData(model)
        data.qpos[:] = scene_pos

        for geom in Constants.OBJECTS:
            model.geom(geom[0]).pos += Constants.MUJOCO_TO_POSE[geom[1]]

        mujoco.mj_step(model, data)
        dr = mujoco.Renderer(model, Constants.CAMERA_HEIGHT, Constants.CAMERA_WIDTH)
        dr.enable_depth_rendering()
        dr.update_scene(data, "frontview")

        simple_depth_img = dr.render()
        simple_depth_img[simple_depth_img >= Constants.THRESHOLD] = 0
        mujoco.mj_step(model, data)

        visible_points_list.append(get_visible_points(simple_depth_img))

        # Add to model for 2nd list
        if any(np.array_equal(mean, sublist) for sublist in estimated_mean):
            geom = allGaussians.add_geom()
            set_values(geom, mean, cls.index(max(cls)))

    model = allGaussiansModelSpec.compile()
    data = mujoco.MjData(model)

    for geom in Constants.OBJECTS:
        model.geom(geom[0]).pos += Constants.MUJOCO_TO_POSE[geom[1]]

    data.qpos[:] = scene_pos
    mujoco.mj_step(model, data)
    dr = mujoco.Renderer(model, Constants.CAMERA_HEIGHT, Constants.CAMERA_WIDTH)
    dr.enable_depth_rendering()
    dr.update_scene(data, "frontview")
    occluded_depth_img = dr.render()
    occluded_depth_img[occluded_depth_img >= Constants.THRESHOLD] = 0
    mujoco.mj_step(model, data)

    # r = mujoco.Renderer(model, Constants.CAMERA_HEIGHT, Constants.CAMERA_WIDTH)
    # r.update_scene(data, "frontview")
    # Image.fromarray(r.render()).show()

    visibilities = []
    for visible_points in visible_points_list:
        visibilities.append(
            calculate_visibility_probability(visible_points, occluded_depth_img)
        )
    print(f"visibilities are {visibilities}")
    return visibilities
