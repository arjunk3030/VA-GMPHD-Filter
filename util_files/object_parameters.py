from collections import defaultdict

THRESHOLD = 9

CAMERA_NAME = "frontview"

ENV_PATH = "environment_assets/EXP2_scene.xml"

TABLE_SIZES = {
    "objects1": [
        0.492,
        1.385,
        1,
        0.7,
        0.4,
        1,
    ],  # x center, y center, z center, x half-length, y half-length, z half-length
    "objects2": [0.1, -0.1, 1, 0.6, 0.4, 1.5],
    "objects3": [-1.565, 0.3, 1, 0.4, 0.7, 1],
}

TABLE_LOCATIONS = {
    "objects1": [0.492, 1.385],
    "objects2": [0.1, -0.1],
    "objects3": [-1.565, 0.3],
}
OBJECT_SETS = {
    "objects1": [
        ["geo1", 2, [-0.02, 0.1], 110],
        ["geo2", 2, [-0.05, -0.1], 180],
        ["geo3", 10, [0.11, 0.12], 0],
        ["geo4", 3, [-0.14, -0.1], 0],
        ["geo5", 9, [0.1275, -0.04], 0],
        ["geo6", 9, [0.1850, -0.07], 35],
        ["geo7", 9, [0.2425, -0.07], 65],
        ["geo8", 13, [-0.13, 0.02], 40],
        ["geo9", 8, [0, 0], 0],
        ["geo10", 13, [0.04, -0.1], 123],
    ],
    "objects2": [
        ["objects2_geo1", 3, [0.01, 0.03], 143],
        ["objects2_geo2", 4, [0.0, 0.13], 0],
        ["objects2_geo3", 12, [0.1875, 0.125], 125],
        ["objects2_geo4", 1, [-0.12, -0.04], -60],
        ["objects2_geo5", 3, [0.09, 0.01], 0],
        ["objects2_geo6", 1, [-0.13, 0.11], 0],
        ["objects2_geo7", 4, [-0.02, -0.1], 0],
        ["objects2_geo8", 4, [0.07, -0.08], 157],
        ["objects2_geo9", 12, [0.21, -0.08], 157],
    ],
    "objects3": [
        ["objects3_geo1", 4, [0.01, 0.03], 143],
        ["objects3_geo2", 3, [0.0, 0.13], 0],
        ["objects3_geo3", 8, [0.2, 0.145], 68],
        ["objects3_geo4", 13, [-0.12, -0.04], -60],
        ["objects3_geo5", 4, [0.09, 0.01], 120],
        ["objects3_geo6", 1, [-0.13, 0.11], 0],
        ["objects3_geo7", 3, [-0.02, -0.1], 0],
        ["objects3_geo8", 3, [0.07, -0.08], 151],
        ["objects3_geo9", 2, [0.21, -0.09], 152],
        ["objects3_geo10", 12, [0.08, 0.35], 23],
    ],
}

FLOOR_HEIGHT = 1.1625

ID_TO_INDEX = {
    2: 0,  # Master Chef Can
    3: 1,  # Cracker Box
    4: 2,  # Sugar Box
    5: 3,  # Tomato Soup Can
    6: 4,  # Mustard Bottle
    7: 5,  # Tuna Fish Can
    8: 6,  # Pudding Box
    9: 7,  # Gelatin Box
    10: 8,  # Potted Meat Can
    11: 9,  # Banana
    19: 10,  # Pitcher Base
    21: 11,  # Bleach Cleanser
    24: 12,  # Bowl
    25: 13,  # Mug
    35: 14,  # Power Drill
    36: 15,  # Wood Block
    37: 16,  # Scissors
    40: 17,  # Large Marker
    51: 18,  # Large Clamp
    52: 19,  # Extra Large Clamp
    61: 20,  # Foam Brick
}

YCB_OBJECT_COUNT = 21
AVAILABLE_OBJECTS = [1, 2, 3, 4, 8, 9, 10, 12, 13, 15]
ROTATION_INVARIANT = [3, 12]


def CLS_TO_MESH(cls):
    if cls not in AVAILABLE_OBJECTS:
        cls = 1
    return f"{str(cls).zfill(3)}"


def CLS_TO_MATERIAL(cls):
    if cls not in AVAILABLE_OBJECTS:
        cls = 1
    return f"{str(cls).zfill(3)}_mtl"


MUJOCO_TO_POSE = defaultdict(
    lambda: [0.0123, 0.0015, 0],
    {
        1: [0.0125, 0.015, 0],
        2: [0.01, 0.0175, 0],
        3: [0.01, -0.082, 0],
        4: [0.0168, 0.022, 0],
        8: [0.032, 0.027, 0],
        9: [0.015, 0, 0],
        10: [0.013, -0.031, 0],
        12: [0.015, 0.048, 0],
        13: [0.02, -0.014, 0],
        15: [-0.02, 0.015, 0],
    },
)
