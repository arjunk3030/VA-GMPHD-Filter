import math
import numpy as np
from Constants import CAMERA_HEIGHT, CAMERA_WIDTH
from scipy.spatial.transform import Rotation as R
import mujoco
import Constants


def compute_rotation_matrix(r):
    camera = r.scene.camera[1]
    # camera_right = np.cross(camera.up, camera.forward)
    # return np.column_stack((camera_right, camera.up, camera.forward))
    forward = np.array(camera.forward, dtype=float)
    up = np.array(camera.up, dtype=float)

    # Calculate right vector
    right = np.cross(-up, forward)

    # # Recalculate up vector to ensure orthogonality
    # up = np.cross(forward, right)

    # Construct 3x3 rotation matrix
    rotation_matrix = np.column_stack((right, -up, forward))

    return rotation_matrix


def compute_translation_matrix(r):
    cameras = r.scene.camera

    positions = np.array([np.array(camera.pos) for camera in cameras])
    avg_position = np.mean(positions, axis=0)

    return avg_position.tolist()


def camera_intrinsic(model):
    fov = model.vis.global_.fovy  # Fixed FOV angle
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT

    fW = 0.5 * height / math.tan(fov * math.pi / 360)
    fH = 0.5 * height / math.tan(fov * math.pi / 360)
    return np.array(((fW, 0, width / 2), (0, fH, height / 2), (0, 0, 1)))


def image_to_camera_coordinates(x, y, depth, K):
    x_normalized = (x - K[0][2]) / K[0][0]
    y_normalized = (y - K[1][2]) / K[1][1]
    x_c = x_normalized * depth
    y_c = y_normalized * depth
    z_c = depth
    return x_c, y_c, z_c


def camera_to_image(x, y, depth, K):
    x_i = (x * K[0][0] / depth) + K[0][2]
    y_i = (y * K[1][1] / depth) + K[1][2]
    return int(x_i), int(y_i)


def camera_to_world(camera_matrix, point_camera):
    return (
        np.dot(camera_matrix["rotation"], point_camera) + camera_matrix["translation"]
    )


def world_to_camera(camera_matrix, point_world):
    return camera_matrix["rotation"].T @ (point_world - camera_matrix["translation"])


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
        ]
    )


def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def calculateAngle(q, rotation):
    new_rot = np.dot(rotation, quaternion_to_rotation_matrix(q))
    new_quat = R.from_matrix(new_rot).as_quat()
    new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
    print("Euler degrees: ", np.degrees(rotation_matrix_to_euler(new_rot)))


# Region growing
def distance_3d(point1, point2, camera_matrix):
    rotation_matrix = camera_matrix["rotation"]
    translation = camera_matrix["translation"]

    point1_world = rotation_matrix @ point1 + translation
    point2_world = rotation_matrix @ point2 + translation

    return np.linalg.norm(point1_world - point2_world)


def near_ground(z):
    threshold = 0.003
    if abs(abs(z) - Constants.FLOOR_HEIGHT) < threshold:
        return True
    if abs(z) < threshold:
        return True
    return False


def createChooseMask(
    depth_image,
    rgbImage,
    camera_matrix,
    seed,
    model,
    tolerance_depth=0.02,
    tolerance_color=100,
):
    rgb_image = np.array(rgbImage)

    height, width = depth_image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    stack = [seed]
    segmented[seed[1], seed[0]] = 255

    while stack:
        x, y = stack.pop()
        current_3d = image_to_camera_coordinates(
            x, y, depth_image[round(y), round(x)], camera_intrinsic(model)
        )
        world_3d = camera_to_world(camera_matrix, current_3d)

        # Skip points near the ground plane
        if near_ground(world_3d[2]):
            continue

        current_color = rgb_image[y, x]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and segmented[ny, nx] == 0:
                neighbor_3d = image_to_camera_coordinates(
                    nx, ny, depth_image[round(ny), round(nx)], camera_intrinsic(model)
                )
                world_3d_neighbor = camera_to_world(camera_matrix, neighbor_3d)

                # Skip points near the ground plane
                if near_ground(world_3d_neighbor[2]):
                    continue

                neighbor_color = rgb_image[ny, nx]

                depth_diff = distance_3d(current_3d, neighbor_3d, camera_matrix)
                color_diff = np.linalg.norm(
                    current_color.astype(float) - neighbor_color.astype(float)
                )

                if depth_diff <= tolerance_depth and color_diff <= tolerance_color:
                    stack.append((nx, ny))
                    segmented[ny, nx] = 255  # Mark the pixel as segmented immediately

    return segmented


def evaluate_mask(mask, bounding_box):
    x_center, y_center, box_width, box_height = bounding_box

    cropped_mask = mask[
        round(y_center - box_height // 2) : round(y_center + box_height // 2),
        round(x_center - box_width // 2) : round(x_center + box_width // 2),
    ]
    return np.sum(cropped_mask == 255)


def region_growing(depth_image, rgbImage, camera_matrix, bounding_box, model):
    x, y, width, height = bounding_box

    initial_points = [
        (x, y),
        (x, y - height // 4),  # above
        (x, y + height // 4),  # below
        (x - width // 4, y),  # left
        (x + width // 4, y),  # right
    ]

    best_score = -1
    best_segmentation = None

    for seed in initial_points:
        segmentation = createChooseMask(
            depth_image,
            rgbImage,
            camera_matrix,
            (round(seed[0]), round(seed[1])),
            model,
        )
        score = evaluate_mask(segmentation, bounding_box)
        if score > best_score:
            best_score = score
            best_segmentation = segmentation

    return best_segmentation


def is_in_image(x, y, width, height):
    return 0 <= x < width and 0 <= y < height
