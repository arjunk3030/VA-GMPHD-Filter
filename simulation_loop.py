import logging

class SimulationLoop:
    """
    Handles stepping through a path, capturing perception, 
    and updating filters for all object sets.
    """

    def __init__(self, robot, perception, filters_by_object, path, actions_by_index, frame_duration, camera_width, camera_height):
        self.robot = robot
        self.perception = perception
        self.filters = filters_by_object
        self.path = path
        self.actions_by_index = actions_by_index
        self.current_step = 0
        self.frame_duration = frame_duration
        self.camera_width = camera_width
        self.camera_height = camera_height

    def step(self):
        """
        Step robot one waypoint forward, capture perception,
        and update all object filters.
        """
        if self.current_step >= len(self.path):
            logging.warning("All waypoints completed")
            return False

        wp = self.path[self.current_step]

        # Move robot
        self.robot.turn_robot(wp.yaw)
        self.robot.move_to_waypoint(self.current_step, self.frame_duration)

        # Capture views
        object_sets = self.actions_by_index.get(self.current_step, [])
        views = self.perception.capture_views(self.current_step, object_sets)

        # Process each view
        for name, view in views.items():
            means, cls, distance = self.perception.process_view(view, name)
            self.filters[name].run_filter(self.robot.data.qpos.copy(), self.robot.data.ctrl.copy(), means, cls, distance, self.camera_width, self.camera_height)

        self.current_step += 1
        return True

    def step_all(self):
        """
        Step through all remaining waypoints.
        """
        while self.current_step < len(self.path):
            self.step()

    def print_filter_results(self):
        for f in self.filters.values():
            means, cls = f.outputFilter()
            for result in means:
                logging.info(result)

    def evaluate_filters(self):
        for f in self.filters.values():
            f.evaluate()
