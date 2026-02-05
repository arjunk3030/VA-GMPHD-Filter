import argparse
import logging
import ssl
import sys
import numpy as np
import torch
import mujoco
import mujoco.viewer as viewer
from util_files.logger_setup import set_debug_mode
from simulation_loop import SimulationLoop
from util_files.TrajectorySettings import PATH, ACTIONS_BY_INDEX
from util_files.object_parameters import CAMERA_NAME, ENV_PATH, OBJECT_SETS, TABLE_LOCATIONS
from util_files.transformation_utils import euler_to_quaternion, quaternion_multiply
from filters.filter_processing import FilterProcessing
from motion import RobotController
from perception import PerceptionManager

ssl._create_default_https_context = ssl._create_stdlib_context


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_objects(model, data, table_name, object_set):
    """
    Initialize objects in Mujoco scene and return ground truth types/positions.
    """
    ground_truth_objs, ground_truth_types = [], []

    for geom in object_set:
        new_quat = euler_to_quaternion(0, 0, geom[3])
        originalZ = model.geom(geom[0]).pos[2]
        model.geom(geom[0]).quat = quaternion_multiply(new_quat, model.geom(geom[0]).quat)
        model.geom(geom[0]).pos = [geom[2][0], geom[2][1], originalZ]
        ground_truth_types.append(geom[1])

        delta_locs = np.array([model.geom(geom[0]).pos[0], model.geom(geom[0]).pos[1]]) + np.array(TABLE_LOCATIONS[table_name])
        ground_truth_objs.append([delta_locs[0], delta_locs[1], geom[3]])

    return ground_truth_types, ground_truth_objs

def main():
    
    parser = argparse.ArgumentParser(description="Run robot simulation with MuJoCo.")
    parser.add_argument(
        "--filter",
        type=str,
        default="VA_GMPHD",
        choices=["VA_GMPHD", "GMPHD", "NOV_GMPHD"],
        help="Filter type to use (options: VA_GMPHD, GMPHD, NOV_GMPHD)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (default: True)")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=True)
    parser.add_argument("--frame-duration", type=float, default=0.5, help="Frame duration in seconds")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera image height")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera image width")
    parser.add_argument("--viewer", action="store_true",help="Enable MuJoCo viewer window",)
    args, x = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + x
    
    set_debug_mode(args.debug)


    # Load Mujoco model
    model = mujoco.MjModel.from_xml_path(ENV_PATH, dict())
    data = mujoco.MjData(model)

    rgb_renderer = mujoco.Renderer(model, args.camera_height, args.camera_width)
    rgb_renderer.update_scene(data)

    depth_renderer = mujoco.Renderer(model, args.camera_height, args.camera_width)
    depth_renderer.enable_depth_rendering()
    depth_renderer.update_scene(data)

    objectDetectionModel = torch.hub.load("ultralytics/yolov5", "custom", path="yolo-for-ycb/best.pt")
    filters = {}
    
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data)
        v = viewer_ctx.__enter__()
    else:
        viewer_ctx = None
        v = None

    try:
        v = None
        robot = RobotController(model, data, v)
        perception = PerceptionManager(model, data, v, rgb_renderer, depth_renderer, CAMERA_NAME, camera_height=args.camera_height, camera_width=args.camera_width, object_detection_model=objectDetectionModel)

        # Initialize objects and filters
        for object_name, object_set in OBJECT_SETS.items():
            types, objs = setup_objects(model, data, object_name, object_set)
            filters[object_name] = FilterProcessing(objs, types, object_name, args.filter)
        robot.step_sim()

        # Initialize simulation loop
        sim_loop = SimulationLoop(robot, perception, filters, PATH, ACTIONS_BY_INDEX, frame_duration=args.frame_duration, camera_width=args.camera_width, camera_height=args.camera_height)

        # Control loop
        while (v is None) or v.is_running():
            cmd = input(
                "Commands:\n"
                "  s - Step robot one step\n"
                "  a - Step through all steps\n"
                "  p - Print filter results\n"
                "  e - Evaluate filters\n"
                "  q - Quit\n"
                "Enter choice: "
            ).strip().lower()

            if cmd in {"q", "quit"}:
                break
            elif cmd in {"s", "step"}:
                sim_loop.step()
            elif cmd in {"a", "step_all"}:
                sim_loop.step_all()
            elif cmd == "p":
                sim_loop.print_filter_results()
            elif cmd == "e":
                sim_loop.evaluate_filters()
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

if __name__ == "__main__":
    main()
