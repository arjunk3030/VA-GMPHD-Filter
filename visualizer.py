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
    # geom.material = CLS_TO_MATERIAL(cls)  # noqa: F821
    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees
    geom.quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
    new_pos = PHDFilterCalculations.asymmetric_to_symmetric_rotation(
        true_center,
        MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    geom.pos = new_pos
    geom.rgba = [0.471, 0.322, 0.769, 0.35]  # light purple color + 0.2 translucency


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
        (1, [-0.020747, 0.013032, 178.64]),
        (1, [-0.027641, -0.1428, 126.86]),
        (3, [0.1882, -0.094235, 106.95]),
        (3, [0.10792, -0.072012, 119.04]),
        (4, [0.094584, 0.032823, 163.03]),
        (4, [0.08651, -0.19876, 150.64]),
        (4, [0.16514, -0.18366, 162.78]),
        (12, [0.27817, 0.028093, 132.16]),
        (12, [0.31764, -0.16946, 137.73]),
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
        viewer.cam.azimuth = 127.66019208715579  # Horizontal angle (degrees)
        viewer.cam.elevation = -67.63532110091747  # Vertical angle (degrees)
        viewer.cam.distance = 3.169448814604967  # Distance from the point of interest

        for object_name, object_set in OBJECT_SETS.items():
            stateUpdates(model, data, object_set)

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
