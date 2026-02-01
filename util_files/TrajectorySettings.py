from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from scipy.interpolate import interp1d

@dataclass
class DetectedObject:
    x: float
    y: float
    z: float
    bbox: List[float]  # [x_center, y_center, width, height]
    cls: int


@dataclass
class View:
    step: int
    rgb: np.ndarray
    depth: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    objects: List[DetectedObject] = field(default_factory=list)

@dataclass
class Waypoint:
    pos: Tuple[float, float] 
    yaw: float


@dataclass
class ViewAction:
    waypoint_idx: int
    targets: List[str]


PATH: List[Waypoint] = [
    Waypoint((0.1, -0.8), 0),
    Waypoint((0.2875, -0.8), 0),
    Waypoint((0.875, -0.8), 0),
    Waypoint((0.875, -0.55), 90),
    Waypoint((0.875, -0.1), 90),
    Waypoint((0.875, 0.3), 90),
    Waypoint((0.875, 0.5), 90),
    Waypoint((0.2875, 0.5), 180),
    Waypoint((0.1, 0.5), 180),
    Waypoint((-0.0875, 0.5), 180),
    Waypoint((-0.675, 0.5), 180),
    Waypoint((-0.675, 0.3), 270),
    Waypoint((-0.675, -0.1), 270),
]

ACTIONS: List[ViewAction] = [
    ViewAction(0, ["objects2"]),
    ViewAction(1, ["objects2"]),
    ViewAction(2, ["objects2"]),
    ViewAction(3, ["objects2", "objects1"]),
    ViewAction(4, ["objects2", "objects1"]),
    ViewAction(5, ["objects2", "objects1"]),
    ViewAction(6, ["objects2", "objects1"]),
    ViewAction(7, ["objects2", "objects1"]),
    ViewAction(8, ["objects2", "objects1"]),
    ViewAction(9, ["objects2", "objects1"]),
    ViewAction(10, ["objects2", "objects1", "objects3"]),
    ViewAction(11, ["objects2", "objects1", "objects3"]),
    ViewAction(12, ["objects2", "objects1", "objects3"]),
]

ACTIONS_OBJECTS1_ONLY: List[ViewAction] = [
    ViewAction(3, ["objects1"]),
    ViewAction(4, ["objects1"]),
    ViewAction(5, ["objects1"]),
    ViewAction(6, ["objects1"]),
    ViewAction(7, ["objects1"]),
    ViewAction(8, ["objects1"]),
    ViewAction(9, ["objects1"]),
    ViewAction(10, ["objects1"]),
    ViewAction(11, ["objects1"]),
    ViewAction(12, ["objects1"]),
]

ACTIONS_OBJECTS2_ONLY: List[ViewAction] = [
    ViewAction(i, ["objects2"]) for i in range(len(PATH))
]

ACTIONS_OBJECTS3_ONLY: List[ViewAction] = [
    ViewAction(10, ["objects3"]),
    ViewAction(11, ["objects3"]),
    ViewAction(12, ["objects3"]),
]

ACTIONS_BY_INDEX = {
    action.waypoint_idx: action.targets
    for action in ACTIONS
}