import torch
from scipy.optimize import linear_sum_assignment

from filters.VA_GMPHD.VA_GMPHD import VA_GaussianMixture

def va_gmphd_training_loss(v_corrected: VA_GaussianMixture, gt_m: torch.Tensor, c=1.0, p=2):
    """
    Differentiable loss for learning p_d and p_v.
    - v_corrected: The mixture AFTER the correction step but BEFORE pruning.
    - gt_m: Ground truth means for the current frame (M, dim).
    """
    n = v_corrected.m.shape[0] 
    m = gt_m.shape[0]
    device = v_corrected.m.device

    if n == 0:
        return torch.tensor(m * (c**p / 2), requires_grad=True, device=device)

    dist_matrix = torch.cdist(v_corrected.m[:, :2], gt_m, p=2)
    dist_matrix_capped = torch.clamp(torch.pow(dist_matrix, p), max=c**p)

    with torch.no_grad():
        cost_np = dist_matrix_capped.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

    target_weights = torch.zeros_like(v_corrected.w)
    target_weights[row_ind] = 1.0
    
    loss_existence = torch.nn.functional.binary_cross_entropy(
        torch.clamp(v_corrected.w, 1e-6, 1-1e-6), 
        target_weights
    )

    if len(row_ind) > 0:
        loc_errors = dist_matrix_capped[row_ind, col_ind]
        loss_loc = torch.mean(v_corrected.w[row_ind] * loc_errors)
    else:
        loss_loc = torch.tensor(0.0, device=device)

    num_missed = m - len(row_ind)
    loss_card = num_missed * (c**p / 2)

    return loss_existence + loss_loc + loss_card