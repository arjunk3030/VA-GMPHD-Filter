# Visibility-Aware Filter (VA-GMPHD)

This repository contains the official code associated with our paper on the **VA-GMPHD (Visibility Aware - Gaussian Mixture Probability Hypothesis Density)** filter. The VA-GMPHD filter enhances object detection and task planning, especially in complex, occluded environments. The project runs on the MuJoCo framework and includes tools to evaluate and compare filtering algorithms on different robot trajectories and object settings.

## Installation

Before running the project, install the dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### MuJoCo Requirements

- **MuJoCo**: Make sure you have **MuJoCo** installed and configured. This project requires only a CPU, not a GPU, but ensure your system meets any additional MuJoCo requirements.
- **mjpython**: The project uses mjpython to interact with MuJoCo.

## Running a Simulation

To run a simulation, simply execute:

```
mjpython main.py
```

A live viewer will appear along with outputs to the console. All plots tracking the cardinality, positions (X, Y, Rotation), and misclassification rates over time will be saved as images.

## Main Files

There are two main files to run the simulation and evaluate the results:

1. **`run_filter.py`**: This file simulates the filtering algorithm based on object trajectories and locations. It is manually operated using the following commands:

```
Commands:
s - Step robot one step
a - Continue stepping through all steps
p - Print filter results
e - Evaluate filters
q - Quit the program
```
Before running the file, please update the following variables/files as needed:
**Debug Mode**: Set the DEBUG_MODE constant (util_files/config_params)
**Environment**: Create the necessary objects set in the environment and set the environment file in ENV_PATH constant (util_files/object_parameters). The pose of each of the objects cannot be set in the environment file but rather in the OBJECT_SETS constant (util_files/object_parameters) following the given template.
**Filter**: Set the filter used via the CURRENT_FILTER constant (util_files/config_params)
**Trajectory**: Set the robot's trajectory in util_files/TrajectorySettings. Enter each of the robot's viewpoints and the object set it should view at that point following the given template

2. **`testing/simulate_results.py`**: This file plots the simulated results, comparing the trajectory and filtering algorithms with the actual results.

## Switching Between Modes and Filters

To switch between debug mode and normal mode, or to switch between filters, modify the configuration in util_files/config_params.py.

## Citations

We borrowed and modified code from **DenseFusion** to integrate into this project:
https://github.com/j96w/DenseFusion/tree/master

The robot.xml file is directly from MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_robot

@inproceedings{rlscale2023rss,
title={Deep RL at Scale: Sorting Waste in Office Buildings with a Fleet of Mobile Manipulators},
author={Alexander Herzog* and Kanishka Rao* and Karol Hausman* and Yao Lu* and Paul Wohlhart\* and Mengyuan Yan and Jessica Lin and Montserrat Gonzalez Arenas and Ted Xiao and Daniel Kappler and Daniel Ho and Jarek Rettinghouse and Yevgen Chebotar and Kuang-Huei Lee and Keerthana Gopalakrishnan and Ryan Julian and Adrian Li and Chuyuan Kelly Fu and Bob Wei and Sangeetha Ramesh and Khem Holden and Kim Kleiven and David Rendleman and Sean Kirmani and Jeff Bingham and Jon Weisz and Ying Xu and Wenlong Lu and Matthew Bennice and Cody Fong and David Do and Jessica Lam and Yunfei Bai and Benjie Holson and Michael Quinlan and Noah Brown and Mrinal Kalakrishnan and Julian Ibarz and Peter Pastor and Sergey Levine
},
booktitle={Robotics: Science and Systems (RSS)},
year={2023}
}

All object meshes are directly from the YCB Video Dataset, a subset of the YCB-Dataset defined here: https://www.ycbbenchmarks.com/object-models/

## Drive Link for Object Weights

Please download the necessary object weights from the following drive link: [Drive Link for Object Weights]()

## License

This project is licensed under the MIT License - see the LICENSE file for details.
