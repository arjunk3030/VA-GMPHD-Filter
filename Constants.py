from collections import defaultdict

TOTAL_STEPS = 8
SPEED = 0.5
DEBUG_MODE = True
THRESHOLD = 9

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
CAMERA_NAME = "frontview"

YCB_OBJECT_COUNT = 21
# OBJECTS = [
#     ["geo1", 4],
#     # ["geo2", 4],
# ]
TABLE_RANGES = {
    "objects1": [0.492, 1.385, 1, 0.7, 0.4, 1],
    "objects2": [0.1, -0.1, 1, 0.6, 0.4, 1.5],  # center
    "objects3": [-1.565, 0.3, 1, 0.4, 0.7, 1],
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
        ["objects3_geo9", 2, [0.21, -0.08], 152],
        ["objects3_geo10", 12, [0.08, 0.35], 23],
    ],
}
FLOOR_NAME = "floor0"
FLOOR_HEIGHT = 1.1625

ID_TO_INDEX = {
    2: 0,  # Master Chef Can
    3: 1,  # Cracker Box CAN USE
    4: 2,  # Sugar Box CAN USE
    5: 3,  # Tomato Soup Can CAN USE
    6: 4,  # Mustard Bottle CAN USE
    7: 5,  # Tuna Fish Can
    8: 6,  # Pudding Box
    9: 7,  # Gelatin Box
    10: 8,  # Potted Meat Can CAN USE
    11: 9,  # Banana CAN USE
    19: 10,  # Pitcher Base CAN USE
    21: 11,  # Bleach Cleanser
    24: 12,  # Bowl CAN USE
    25: 13,  # Mug CAN USE
    35: 14,  # Power Drill
    36: 15,  # Wood Block
    37: 16,  # Scissors
    40: 17,  # Large Marker
    51: 18,  # Large Clamp
    52: 19,  # Extra Large Clamp
    61: 20,  # Foam Brick
}


def CLS_TO_MESH(cls):
    if cls not in VALID_INDICES:
        cls = 1
    return f"{str(cls).zfill(3)}"


def CLS_TO_MATERIAL(cls):
    if cls not in VALID_INDICES:
        cls = 1
    return f"{str(cls).zfill(3)}_mtl"


VALID_INDICES = [1, 2, 3, 4, 8, 9, 10, 12, 13, 15]

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
