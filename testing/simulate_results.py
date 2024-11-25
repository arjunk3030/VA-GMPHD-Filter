import logging
from mujoco import mjtGeom

# from ModelProcessor import ModelProcessor
import mujoco
import mujoco.viewer as viewer
import numpy as np
import ssl
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util_files.transformation_utils import (
    asymmetric_to_symmetric_rotation,
    euler_to_quaternion,
)
from util_files.config_params import CAMERA_HEIGHT, CAMERA_WIDTH
import filters.calculate_visibility as calculate_visibility
from util_files.object_parameters import (
    CLS_TO_MATERIAL,
    CLS_TO_MESH,
    FLOOR_HEIGHT,
    MUJOCO_TO_POSE,
    OBJECT_SETS,
)

ssl._create_default_https_context = ssl._create_stdlib_context


def createGeom(scene: mujoco.MjvScene, location, rgba_given):
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.0075, 0.0075, 0.0075],
        pos=np.array([location[0], location[1], location[2]]),
        mat=np.eye(3).flatten(),
        rgba=rgba_given,
    )


def set_values(geom, mean, cls):
    geom.type = mjtGeom.mjGEOM_MESH
    geom.meshname = CLS_TO_MESH(cls)
    # geom.material = CLS_TO_MATERIAL(cls)  # noqa: F821
    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees
    geom.quat = euler_to_quaternion(0, 0, mean[2])
    new_pos = asymmetric_to_symmetric_rotation(
        true_center,
        MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    geom.pos = new_pos
    geom.rgba = [0.471, 0.322, 0.769, 0.35]  # light purple color + 0.2 translucency


def stateUpdates(model, data, object_set):
    for geom in object_set:
        newQuat = euler_to_quaternion(0, 0, geom[3])
        originalZ = model.geom(geom[0]).pos[2]
        model.geom(geom[0]).quat = calculate_visibility.quaternion_multiply(
            newQuat, model.geom(geom[0]).quat
        )
        model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    list2 = [
        (2, [0.43704, 1.2792, 179.96]),
        (2, [0.4719, 1.4902, 113.8]),
        (3, [0.35267, 1.283, 152.27]),
        (9, [0.6863, 1.2844, 111.23]),
        (10, [0.60519, 1.4904, -6.5589]),
        (13, [0.52715, 1.2782, 153.5]),
        (20, [0.35734, 1.4, 148.58]),
    ]

    # list3 = [
    #     (1, [-0.028176, 0.009926, 177.53]),
    #     (1, [-0.024711, -0.14291, -56.079]),
    #     (12, [0.28332, 0.029457, 178.1]),
    # ]
    # NEW NEW ONE

    spec1 = mujoco.MjSpec()
    spec1.from_file("environment_assets/EXP2_scene.xml")

    object_body = spec1.worldbody.add_body()
    # for result in list1:
    #     object_body.pos = [0, 0, FLOOR_HEIGHT]
    #     g = object_body.add_geom()
    #     set_values(g, result[1], result[0])
    # for result in list2:
    #     object_body.pos = [0, 0, FLOOR_HEIGHT]
    #     g = object_body.add_geom()
    #     set_values(g, result[1], result[0])
    # for result in list3:
    #     object_body.pos = [0, 0, FLOOR_HEIGHT]
    #     g = object_body.add_geom()
    #     set_values(g, result[1], result[0])

    # Set up Mujoco model
    model = spec1.compile()
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]  # Point of interest (x, y, z)
        viewer.cam.azimuth = 127.66019208715579  # Horizontal angle (degrees)
        viewer.cam.elevation = -67.63532110091747  # Vertical angle (degrees)
        viewer.cam.distance = 3.169448814604967  # Distance from the point of interest

        # viewer.cam.lookat[:] = [0.0, 0.0, 0.0]  # Point of interest (x, y, z)
        # viewer.cam.azimuth = -14.808629587156048  # Horizontal angle (degrees)
        # viewer.cam.elevation = -39.8995126146786  # Vertical angle (degrees)
        # viewer.cam.distance = 2.864629740942742  # Distance from the point of interest

        # for object_name, object_set in OBJECT_SETS.items():
        #     stateUpdates(model, data, object_set)

        scene_option = mujoco.MjvOption()
        scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        mujoco.mj_step(model, data)
        viewer.sync()

        while viewer.is_running():
            x = input("Click s to step robot, click c to step through all: ")
            if x == "q" or x == "Q":
                # logger.info(f"Lookat: {viewer.cam.lookat}")
                # logger.info(f"Azimuth: {viewer.cam.azimuth}")
                # logger.info(f"Elevation: {viewer.cam.elevation}")
                # logger.info(f"Distance: {viewer.cam.distance}")
                viewer.close()
                break
