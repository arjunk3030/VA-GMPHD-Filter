import math
import numpy as np
from util_files.keys import ROTATION_KEY, TRANSLATION_KEY
from util_files.object_parameters import FLOOR_HEIGHT, TABLE_SIZES
from util_files.transformation_utils import quaternion_to_rotation_matrix
from logger_setup import logger


def compute_rotation_matrix(r):
    camera = r.scene.camera[1]
    forward = np.array(camera.forward, dtype=float)

    up = np.array(camera.up, dtype=float)
    right = np.cross(-up, forward)
    rotation_matrix = np.column_stack((right, -up, forward))

    return rotation_matrix


def compute_translation_matrix(r):
    cameras = r.scene.camera

    positions = np.array([np.array(camera.pos) for camera in cameras])
    avg_position = np.mean(positions, axis=0)

    return avg_position.tolist()


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
        np.dot(camera_matrix[ROTATION_KEY], point_camera)
        + camera_matrix[TRANSLATION_KEY]
    )


def world_to_camera(camera_matrix, point_world):
    return camera_matrix[ROTATION_KEY].T @ (
        point_world - camera_matrix[TRANSLATION_KEY]
    )


def rotation_matrix_to_z_rotation(R):
    """
    Converts a 3x3 rotation matrix to the z-axis rotation angle in degrees.

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    float: The z-axis rotation angle in degrees.
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Calculate the z-axis rotation angle
    z_rotation = math.atan2(r21, r11)
    z_rotation_degrees = math.degrees(z_rotation)

    return z_rotation_degrees


def calculateAngle(q, rotation):
    q = [q[0], q[1], q[2], q[3]]
    new_rot = np.dot(rotation, quaternion_to_rotation_matrix(q))
    last_rotation = rotation_matrix_to_z_rotation((new_rot))
    logger.info(f"Rotation on the z axis is: f{last_rotation}")
    return last_rotation


# Region growing
def distance_3d(point1, point2, camera_matrix):
    rotation_matrix = camera_matrix[ROTATION_KEY]
    translation = camera_matrix[TRANSLATION_KEY]

    point1_world = rotation_matrix @ point1 + translation
    point2_world = rotation_matrix @ point2 + translation

    return np.linalg.norm(point1_world - point2_world)


def near_ground(z):
    threshold = 0.006
    if abs(abs(z) - FLOOR_HEIGHT) < threshold:
        return True
    if abs(z) < threshold:
        return True
    return False


def createChooseMask(
    depth_image,
    rgbImage,
    camera_matrix,
    seed,
    intrinsic,
    tolerance_depth=0.01,
    tolerance_color=130,
):
    rgb_image = np.array(rgbImage)

    height, width = depth_image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    stack = [seed]
    segmented[seed[1], seed[0]] = 255

    while stack:
        x, y = stack.pop()
        current_3d = image_to_camera_coordinates(
            x, y, depth_image[round(y), round(x)], intrinsic
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
                    nx, ny, depth_image[round(ny), round(nx)], intrinsic
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


def is_point_in_3d_box(point_2d, object_set, camera_matrix, depth_image, intrinsic):
    center_x, center_y, center_z, radius_x, radius_y, radius_z = TABLE_SIZES[object_set]

    # Convert the 2D point to camera coordinates
    point_3d_camera = image_to_camera_coordinates(
        point_2d[0],
        point_2d[1],
        depth_image[round(point_2d[1]), round(point_2d[0])],
        intrinsic,
    )

    # Convert the camera coordinates to world coordinates
    point_3d_world = camera_to_world(camera_matrix, point_3d_camera)

    # Check if the point lies within the 3D box
    return (
        (center_x - radius_x <= point_3d_world[0] <= center_x + radius_x)
        and (center_y - radius_y <= point_3d_world[1] <= center_y + radius_y)
        and (center_z - radius_z <= point_3d_world[2] <= center_z + radius_z)
    )


def region_growing(depth_image, rgbImage, camera_matrix, bounding_box, intrinsic):
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
            intrinsic,
        )
        score = evaluate_mask(segmentation, bounding_box)
        if score > best_score:
            best_score = score
            best_segmentation = segmentation

    return best_segmentation


def is_in_image(x, y, width, height):
    return 0 <= x < width and 0 <= y < height
