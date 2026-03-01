import numpy as np
from filters.VA_GMPHD.VA_GMPHD import clutter_intensity_function
from util_files.object_parameters import YCB_OBJECT_COUNT


import torch

def VA_GMPHD_model(device=torch.device("cpu")):
    model = {}

    # Sampling time, time step duration
    T_s = 1.0
    model["T_s"] = T_s
    model["nObj"] = YCB_OBJECT_COUNT

    # Surveillance region
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -2.0, 3.0
    model["surveillance_region"] = torch.tensor([[x_min, x_max], [y_min, y_max]], device=device)

    # Transition matrix (Identity 3x3)
    I_3 = torch.eye(3, device=device)

    # Process noise covariance matrix
    sigma_w_xy = 0.03
    sigma_w_z = 60.0
    model["Q"] = torch.diag(torch.tensor([sigma_w_xy**2, sigma_w_xy**2, sigma_w_z**2], device=device))

    model["birth_w"] = torch.tensor(0.6, device=device)
    model["birth_P"] = torch.diag(torch.tensor([0.0375, 0.0375, 60.0], device=device))

    # MEASUREMENT MODEL
    model["specs"] = torch.tensor([1.0, 0.4, 0.6, 0.2], device=device)
    model["alpha"] = 0.9

    # Measurement matrix
    model["H"] = I_3
    
    # Measurement noise covariance matrix
    sigma_v_xy = 0.05
    sigma_v_z = 50.0
    model["R"] = torch.diag(torch.tensor([sigma_v_xy**2, sigma_v_xy**2, sigma_v_z**2], device=device))

    # The reference to clutter intensity function
    model["lc"] = 0.45
    
    # Updated lambda to ensure it uses the torch-based clutter function
    model["clutt_int_fun"] = lambda z: clutter_intensity_function(
        z, model["lc"], model["surveillance_region"]
    )

    model["A"] = 0.11
    # Pruning and merging parameters
    model["T"] = 0.19
    model["U"] = 0.09
    model["Jmax"] = 60

    return model