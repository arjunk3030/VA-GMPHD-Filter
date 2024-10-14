import numpy as np
from filters.VA_GMPHD.VA_GMPHD import clutter_intensity_function
from util_files.object_parameters import YCB_OBJECT_COUNT


def VA_GMPHD_model():
    model = {}

    # Sampling time, time step duration
    T_s = 1.0
    model["T_s"] = T_s
    model["nObj"] = YCB_OBJECT_COUNT

    # Surveillance region
    x_min = -4
    x_max = 4
    y_min = -2
    y_max = 3
    model["surveillance_region"] = np.array([[x_min, x_max], [y_min, y_max]])

    # Transition matrix
    I_3 = np.eye(3)

    # Process noise covariance matrix
    Q = (T_s**2) * I_3
    # Standard deviation of the process noise
    sigma_w_xy = 0.03  # Standard deviation for x and y
    sigma_w_z = 60
    Q = np.diag([sigma_w_xy**2, sigma_w_xy**2, sigma_w_z**2])
    model["Q"] = Q

    model["birth_w"] = 0.6
    model["birth_P"] = np.diag([0.0375, 0.0375, 60])

    # MEASUREMENT MODEL
    # Probability of detection
    model["specs"] = [1, 0.4, 0.6, 0.2]  # 0.5
    model["alpha"] = 0.875

    # Measurement matrix z = Hx + v = N(z; Hx, R)
    model["H"] = I_3  # Since we are now measuring (x, y, z)
    # Measurement noise covariance matrix
    sigma_v_xy = 0.05  # Standard deviation for measurement noise in x and y
    sigma_v_z = 50  # Larger standard deviation for z due to higher measurement noise
    model["R"] = np.diag([sigma_v_xy**2, sigma_v_xy**2, sigma_v_z**2])

    # The reference to clutter intensity function
    model["lc"] = 0.45
    model["clutt_int_fun"] = lambda z: clutter_intensity_function(
        z, model["lc"], model["surveillance_region"]
    )

    model["A"] = 0.11
    # Pruning and merging parameters
    model["T"] = 0.19
    model["U"] = 0.09
    model["Jmax"] = 60

    return model
