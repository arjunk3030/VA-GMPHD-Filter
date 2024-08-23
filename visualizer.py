import logging
import random
from Constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CLS_TO_MATERIAL,
    CLS_TO_MESH,
    DEBUG_MODE,
    DURATION_IN_SECONDS,
    FLOOR_HEIGHT,
    MUJOCO_TO_POSE,
    OBJECTS,
    THRESHOLD,
    TOTAL_STEPS,
)
from DenseProcessor import DenseProcessor
from TrajectorySettings import TRAJECTORY_PATH
from filter_init import PhdFilter
from mujoco import mjtGeom

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
import PHDFilterCalculations
from ultralytics import YOLO
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
    geom.meshname = CLS_TO_MESH[cls]
    # quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
    # geom.pos = PHDFilterCalculations.compute_mujoco_position(
    #     [mean[0], mean[1], 0], MUJOCO_TO_POSE[cls], mean[2]
    # )

    true_center = np.array([mean[0], mean[1], 0])
    euler_angles = np.array([0, 0, mean[2]])  # in degrees

    new_pos = PHDFilterCalculations.asymmetric_to_symmetric_rotation(
        true_center,
        MUJOCO_TO_POSE[cls],
        euler_angles,
    )
    geom.pos = new_pos
    geom.quat = PointEstimation.euler_to_quaternion(0, 0, mean[2])
    print(f"this is {geom.quat}")
    geom.rgba = [0.694, 0.612, 0.851, 0.2]  # light purple color + 0.2 translucency


def stateUpdates(model, scene, data):
    ground_truth = []
    for geom in OBJECTS:
        createGeom(
            viewer.user_scn,
            model.geom(geom[0]).pos,
            [1, 1, 1, 1],
        )
        model.geom(geom[0]).pos += MUJOCO_TO_POSE[geom[1]]
        ground_truth.append(
            [
                geom[1],
                round(model.geom(geom[0]).pos[0], 3),
                round(model.geom(geom[0]).pos[1], 3),
            ]
        )
        createGeom(
            viewer.user_scn,
            model.geom(geom[0]).pos,
            [random.random(), random.random(), random.random(), 1],
        )

    mujoco.mj_step(model, data)
    return ground_truth


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
    #     (12, [0.25, -0.037, -100000]),
    #     (12, [0.188, 0.129, 2290]),
    # ]
    list = [
        (1, [-0.062205, 0.0032801, 111.2]),
        (1, [-0.11404, 0.13079, 172.81]),
        (3, [0.028196, 0.028351, 19.608]),
        (3, [0.10846, 0.0093407, 98.463]),
        (4, [0.090667, -0.10706, 162.46]),
        (4, [-0.021967, -0.10044, 57.805]),
        (4, [0.010096, 0.15644, 172.91]),
        (12, [0.26053, -0.035342, 93.804]),
        (12, [0.17712, 0.12658, 136.44]),
    ]
    spec1 = mujoco.MjSpec()
    spec1.from_file("google_robot/EXP2_scene.xml")

    object_body = spec1.worldbody.add_body()
    for result in list:
        object_body.pos = [0, 0, FLOOR_HEIGHT]
        g = object_body.add_geom()
        set_values(g, result[1], result[0])

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
        ground_truth = stateUpdates(model, viewer.user_scn, data)
        ground_truth.sort(key=lambda x: x[0])
        print(ground_truth)

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
