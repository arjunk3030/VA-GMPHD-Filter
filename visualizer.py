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
    geom.rgba = [0.694, 0.612, 0.851, 0.6]  # light purple color + 0.2 translucency


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

    # list = [
    #     (1, [-0.088112, -0.039976, 125.88]),
    #     (3, [0.086133, -0.038496, 0]),
    #     (3, [0.031374, -0.1188, 0]),
    #     (4, [0.01143, 0.15572, 0]),
    #     (12, [0.1875, 0.125, 0]),
    #     (15, [-0.14054, 0.14164, 0]),
    # ]

    # list = [
    #     (1, [-0.062205, 0.0032801, 111.2]),
    #     (1, [-0.11404, 0.13079, 172.81]),
    #     (3, [0.028196, 0.028351, 19.608]),
    #     (3, [0.10846, 0.0093407, 98.463]),
    #     (4, [0.090667, -0.10706, 162.46]),
    #     (4, [-0.021967, -0.10044, 57.805]),
    #     (4, [0.010096, 0.15644, 172.91]),
    #     (12, [0.26053, -0.035342, 93.804]),
    #     (12, [0.17712, 0.12658, 136.44]),
    # ]

    # NEW ONES

    # list3 = [
    #     (1, [-1.6852, 0.41406, 19.658]),
    #     (2, [-1.3574, 0.2295, 66.175]),
    #     (2, [-1.4199, 0.3985, -119.74]),
    #     (3, [-1.5587, 0.42421, 44.796]),
    #     (3, [-1.5315, 0.51156, -5.7506]),
    #     (3, [-1.496, 0.38755, 93.559]),
    #     (3, [-1.5537, 0.36444, 142.16]),
    #     (3, [-1.5855, 0.19956, 58.32]),
    #     (3, [-1.4953, 0.21447, 163.2]),
    #     (4, [-1.4767, 0.3086, 138.79]),
    #     (8, [-1.3656, 0.44026, 107.48]),
    #     (12, [-1.4848, 0.632, 67.622]),
    #     (13, [-1.4212, 0.52002, 56.206]),
    # ]

    # list1 = [
    #     (2, [0.4421, 1.2835, 177.19]),
    #     (2, [0.47167, 1.4875, 122.35]),
    #     (2, [0.36924, 1.3048, 144.77]),
    #     (2, [0.49861, 1.4705, -64.316]),
    #     (3, [0.35444, 1.2875, 147.19]),
    #     (3, [0.39262, 1.2812, 31.614]),
    #     (9, [0.70748, 1.2855, 58.633]),
    #     (9, [0.66537, 1.2661, 45.672]),
    #     (9, [0.51796, 1.3005, 39.246]),
    #     (9, [0.56943, 1.2882, -36.278]),
    #     (10, [0.60503, 1.4807, -15.027]),
    #     (10, [0.56534, 1.3969, 35.622]),
    #     (10, [0.57825, 1.4671, -122.54]),
    #     (13, [0.53263, 1.2752, 72.567]),
    #     (13, [0.35746, 1.4003, 164.91]),
    #     (13, [0.49432, 1.2827, -77.695]),
    # ]

    # list2 = [
    #     (1, [-0.026055, 0.0071369, 178.61]),
    #     (1, [-0.026108, -0.14297, 121.48]),
    #     (3, [0.18528, -0.093457, 57.043]),
    #     (3, [0.088566, -0.066259, 154.6]),
    #     (4, [0.10132, 0.026215, 177.72]),
    #     (4, [0.16733, -0.18138, 156.55]),
    #     (4, [0.087458, -0.2014, 178.74]),
    #     (12, [0.27619, 0.021806, 131.89]),
    #     (12, [0.3095, -0.17581, 155.14]),
    # ]

    # NEW ONE
    # list1 = [
    #     (1, [-1.4252, 0.25454, 60.401]),
    #     (1, [-1.693, 0.41638, 2.6407]),
    #     (2, [-1.3606, 0.22582, 44.287]),
    #     (3, [-1.4921, 0.21597, 141.4]),
    #     (3, [-1.5614, 0.42823, 46.33]),
    #     (3, [-1.5852, 0.19855, 63.208]),
    #     (4, [-1.4811, 0.31322, 173.89]),
    #     (8, [-1.3499, 0.44586, 118.93]),
    #     (12, [-1.4764, 0.64687, 118.02]),
    # ]

    # OBJECTS_2 = [
    #     (1, [-0.027135, -0.14355, 130.81]),
    #     (1, [-0.017359, 0.014527, 174.24]),
    #     (3, [0.18817, -0.094172, 106.88]),
    #     (3, [0.1078, -0.071339, 118.96]),
    #     (4, [0.094279, 0.032589, 163.03]),
    #     (4, [0.085691, -0.19963, 160.01]),
    #     (4, [0.16512, -0.18357, 161.38]),
    #     (12, [0.27815, 0.028257, 132.16]),
    #     (12, [0.31945, -0.1694, 137.73]),
    # ]
    # OBJECTS_3 = [
    #     (1, [-1.693, 0.41487, 1.9099]),
    #     (2, [-1.3512, 0.23236, 165.54]),
    #     (3, [-1.564, 0.42912, 34.424]),
    #     (3, [-1.5824, 0.1991, 45.047]),
    #     (3, [-1.3857, 0.1763, 25.959]),
    #     (4, [-1.4731, 0.30944, 178.87]),
    #     (8, [-1.3509, 0.4442, 104.63]),
    #     (12, [-1.4828, 0.6477, 67.763]),
    # ]
    list2 = [
        (2, [0.4368, 1.2798, 179.96]),
        (2, [0.47536, 1.4911, 113.8]),
        (3, [0.35025, 1.2819, 153.5]),
        (9, [0.65988, 1.2958, 22.532]),
        (9, [0.72113, 1.2784, 171.01]),
        (10, [0.60416, 1.4889, -3.0239]),
        (13, [0.5293, 1.2765, 150.46]),
        (13, [0.35738, 1.4009, 148.65]),
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
                viewer.close()
                break
