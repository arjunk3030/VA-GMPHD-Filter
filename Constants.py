from collections import defaultdict

TOTAL_STEPS = 8
DURATION_IN_SECONDS = 6
DEBUG_MODE = True
THRESHOLD = 9
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

YCB_OBJECT_COUNT = 21
OBJECTS = [
    ["geo1", 3],
    ["geo2", 4],
    ["geo3", 12],
    ["geo4", 1],
    ["geo5", 3],
    ["geo6", 1],
    ["geo6", 4],
    ["geo7", 4],
    ["geo9", 12],
]
FLOOR_NAME = "floor0"
FLOOR_HEIGHT = 0.76

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


# Create a defaultdict with a default value
CLS_TO_MESH = defaultdict(
    lambda: "006_mustard_bottle",
    {
        1: "003_cracker_box",
        3: "005_tomato_soup_can",
        4: "006_mustard_bottle",
        12: "024_bowl",
        15: "036_wood_block",
    },
)


CLS_TO_MATERIAL = defaultdict(
    lambda: "006_mtl",
    {
        1: "003_mtl",
        3: "005_mtl",
        4: "006_mtl",
        12: "024_mtl",
        15: "036_mtl",
    },
)

MUJOCO_TO_POSE = defaultdict(
    lambda: [0.0168, 0.022, 0],
    {
        1: [0.0125, 0.015, 0],
        15: [-0.02, 0.015, 0],
        3: [0.01, -0.082, 0],
        12: [0.015, 0.048, 0],
        4: [0.0168, 0.022, 0],
    },
)
