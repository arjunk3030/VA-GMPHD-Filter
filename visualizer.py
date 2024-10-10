import logging
import random
from Constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CLS_TO_MATERIAL,
    CLS_TO_MESH,
    FLOOR_HEIGHT,
    MUJOCO_TO_POSE,
    OBJECT_SETS,
)
from mujoco import mjtGeom

# from ModelProcessor import ModelProcessor
import mujoco
import mujoco.viewer as viewer
import numpy as np
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import ssl
import PointEstimation
import PHDFilterCalculations
import torch

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
    geom.material = CLS_TO_MATERIAL(cls)  # noqa: F821
    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees
    geom.quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
    new_pos = PHDFilterCalculations.asymmetric_to_symmetric_rotation(
        true_center,
        MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    geom.pos = new_pos
    # geom.rgba = [0.471, 0.322, 0.769, 0.35]  # light purple color + 0.2 translucency


# def set_values(geom, mean, cls):
#     geom.type = mjtGeom.mjGEOM_MESH
#     geom.meshname = CLS_TO_MESH(cls)
#     # quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
#     # geom.pos = PHDFilterCalculations.compute_mujoco_position(
#     #     [mean[0], mean[1], 0], MUJOCO_TO_POSE[cls], mean[2]
#     # )

#     true_center = np.array([mean[0], mean[1], 0])
#     euler_angles = np.array([0, 0, mean[2]])  # in degrees

#     new_pos = PHDFilterCalculations.asymmetric_to_symmetric_rotation(
#         true_center,
#         MUJOCO_TO_POSE[cls],
#         euler_angles,
#     )
#     geom.pos = new_pos
#     geom.quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
#     print(f"this is {geom.quat}")
#     geom.rgba = [0.694, 0.612, 0.851, 0.2]  # light purple color + 0.2 translucency


def stateUpdates(model, data, object_set):
    for geom in object_set:
        newQuat = PointEstimation.euler_to_quaternion(0, 0, geom[3])
        originalZ = model.geom(geom[0]).pos[2]
        model.geom(geom[0]).quat = PHDFilterCalculations.quaternion_multiply(
            newQuat, model.geom(geom[0]).quat
        )
        model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]


# def stateUpdates(model, scene, data):
#     ground_truth = []
# for geom in OBJECTS:
#     createGeom(
#         viewer.user_scn,
#         model.geom(geom[0]).pos,
#         [1, 1, 1, 1],
#     )
#     model.geom(geom[0]).pos += MUJOCO_TO_POSE[geom[1]]
#     ground_truth.append(
#         [
#             geom[1],
#             round(model.geom(geom[0]).pos[0], 3),
#             round(model.geom(geom[0]).pos[1], 3),
#         ]
#     )
#     createGeom(
#         viewer.user_scn,
#         model.geom(geom[0]).pos,
#         [random.random(), random.random(), random.random(), 1],
#     )

# mujoco.mj_step(model, data)
# return ground_truth


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    list2 = [
        (1, [-1.4259, 0.23723, 82.601]),
        (2, [-1.3515, 0.23167, 165.54]),
        (3, [-1.5652, 0.42815, 34.424]),
        (3, [-1.5824, 0.1991, 45.047]),
        (3, [-1.3848, 0.17761, 37.771]),
        (8, [-1.348, 0.44467, 104.63]),
        (12, [-1.4817, 0.64852, 67.763]),
    ]

    # list3 = [
    #     (1, [-0.028176, 0.009926, 177.53]),
    #     (1, [-0.024711, -0.14291, -56.079]),
    #     (12, [0.28332, 0.029457, 178.1]),
    # ]
    # NEW NEW ONE

    spec1 = mujoco.MjSpec()
    spec1.from_file("google_robot/EXP2_scene.xml")

    object_body = spec1.worldbody.add_body()
    # for result in list1:
    #     object_body.pos = [0, 0, FLOOR_HEIGHT]
    #     g = object_body.add_geom()
    #     set_values(g, result[1], result[0])
    for result in list2:
        object_body.pos = [0, 0, FLOOR_HEIGHT]
        g = object_body.add_geom()
        set_values(g, result[1], result[0])
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
        viewer.cam.azimuth = -14.808629587156048  # Horizontal angle (degrees)
        viewer.cam.elevation = -39.8995126146786  # Vertical angle (degrees)
        viewer.cam.distance = 2.864629740942742  # Distance from the point of interest
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
                # print(f"Lookat: {viewer.cam.lookat}")
                # print(f"Azimuth: {viewer.cam.azimuth}")
                # print(f"Elevation: {viewer.cam.elevation}")
                # print(f"Distance: {viewer.cam.distance}")
                viewer.close()
                break
