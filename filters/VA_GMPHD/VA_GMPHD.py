import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any
import torch


def multivariate_gaussian(x: torch.Tensor, m: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Multivatiate Gaussian Distribution

    :param x: vector
    :param m: distribution mean vector
    :param P: Covariance matrix
    :return: probability density function at x
    """
    detP = torch.linalg.det(P)
    invP = torch.linalg.inv(P)
    return multivariate_gaussian_predefined_det_and_inv(x, m, detP, invP)

def multivariate_gaussian_predefined_det_and_inv(
    x: torch.Tensor, m: torch.Tensor, detP: torch.Tensor, invP: torch.Tensor
) -> torch.Tensor:
    """
    Multivariate Gaussian Distribution with provided determinant and inverse of the Gaussian mixture.
    Useful in case when we already have precalculted determinant and inverse of the covariance matrix.
    :param x: vector
    :param m: distribution mean
    :param detP: determinant of the covariance matrix
    :param invP: inverse of the covariance matrix
    :return: probability density function at x
    """
    dim = x.shape[0]
    first_part = 1 / (torch.pow(2 * torch.pi, dim / 2.0) * torch.sqrt(detP))
    diff = x - m
    second_part = -0.5 * diff @ invP @ diff
    return first_part * torch.exp(second_part)

def clutter_intensity_function(z: torch.Tensor, lc: float, surveillance_region: torch.Tensor):
    """
    Clutter intensity function, with the uniform distribution through the surveillance region, pg. 8
    in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark.
    :param z:
    :param lc: average number of false detections per time step
    :param surveillance_region: np.ndarray of shape (number_dimensions, 2) giving the range(min and max) for each
                                dimension
    """
    if (
        surveillance_region[0][0] <= z[0] <= surveillance_region[0][1]
        and surveillance_region[1][0] <= z[1] <= surveillance_region[1][1]
    ):
        area = (surveillance_region[0][1] - surveillance_region[0][0]) * \
               (surveillance_region[1][1] - surveillance_region[1][0])
        return torch.tensor(lc / area, device=z.device)
    else:
        return torch.tensor(1e-10, device=z.device)

class VA_GaussianMixture:
    def __init__(
        self,
       w: torch.Tensor,
        m: torch.Tensor,
        P: torch.Tensor,
        cls: torch.Tensor,
    ):
        """
        The Gaussian mixture class

        :param w: list of scalar weights
        :param m: list of np.ndarray means
        :param P: list of np.ndarray covariance matrices

        Note that constructor creates detP and invP variables which can be used instead of P list, for covariance matrix
        determinant and inverse. These lists cen be initialized with assign_determinant_and_inverse function, and
        it is useful in case we already have precalculated determinant and inverse earlier.
        """
        self.w = w
        self.m = m
        self.P = P
        self.cls = cls
        self.detP = None
        self.invP = None

    def set_covariance_determinant_and_inverse_list(
        self, detP: torch.Tensor, invP: torch.Tensor
    ):
        """
        For each Gaussian component, provide the determinant and the covariance inverse
        :param detP: list of determinants for each Gaussian component in the mixture
        :param invP: list of covariance inverses for each Gaussian component in the mixture
        """
        self.detP = detP
        self.invP = invP

    def mixture_value(self, x: torch.Tensor):
        """
        Gaussian Mixture function for the given vector x
        """
        sum = torch.tensor(0.0, device=x.device)
        if self.detP is None:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian_predefined_det_and_inv(
                    x, self.m[i], self.detP[i], self.invP[i]
                )
        return sum

    def mixture_single_component_value(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Single Gaussian Mixture component value for the given vector
        :param x: vector
        :param i: index of the component
        :returns: probability density function at x, multiplied with the component weght at the index i
        """
        if self.detP is None:
            return self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            return self.w[i] * multivariate_gaussian_predefined_det_and_inv(
                x, self.m[i], self.detP[i], self.invP[i]
            )

    def mixture_component_values_list(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sometimes it is useful to have the value of each component multiplied with its weight
        :param x: vector
        :return: List[np.float64]:
        List of components values at x, multiplied with their weight.
        """
        val = []
        x_2d = x[:2]  # Use only the first two elements of x

        if self.detP is None:
            for i in range(len(self.w)):
                m_2d = self.m[i][
                    :2
                ]  # Use only the first two elements of the mean vector
                P_2x2 = self.P[i][
                    :2, :2
                ]  # Extract the 2x2 submatrix of the covariance matrix
                val.append(self.w[i] * multivariate_gaussian(x_2d, m_2d, P_2x2))
        else:
            for i in range(len(self.w)):
                m_2d = self.m[i][
                    :2
                ]  # Use only the first two elements of the mean vector
                P_2x2 = self.P[i][
                    :2, :2
                ]  # Extract the 2x2 submatrix of the covariance matrix
                invP_2x2 = self.invP[i][
                    :2, :2
                ]  # Extract the 2x2 submatrix of the inverse covariance matrix
                val.append(
                    self.w[i]
                    * multivariate_gaussian_predefined_det_and_inv(
                        x_2d, m_2d, self.detP[i], invP_2x2
                    )
                )
        return torch.stack(val)

    def copy(self):
        w = self.w.clone()
        m = self.m.clone()
        P = self.P.clone()
        cls = self.cls.clone()
        return VA_GaussianMixture(w, m, P, cls)


def get_matrices_inverses(P_tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(P_tensor)

def get_matrices_determinants(P_tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.det(P_tensor)

def thinning_and_displacement(v: VA_GaussianMixture, p: torch.Tensor, F: torch.Tensor, Q: torch.Tensor):
    w = v.w * p
    m = v.m @ F.T
    P = F @ v.P @ F.T + Q
    return VA_GaussianMixture(w, m, P, v.cls)

def just_displacement(v: VA_GaussianMixture, F: torch.Tensor, Q: torch.Tensor):
    m = v.m @ F.T
    P = F @ v.P @ F.T + Q
    return VA_GaussianMixture(v.w, m, P, v.cls)

def spread_convariance(v: VA_GaussianMixture, Q: torch.Tensor):
    w = v.w.clone()
    m = v.m.clone()
    P = v.P + Q
    return VA_GaussianMixture(w, m, P, v.cls)


class VA_GMPHD:
    def __init__(self, model: Dict[str, Any]):
        """
        The Gaussian Mixture Probability Hypothesis Density filter implementation.
        "The Gaussian mixture probability hypothesis density filter" by Vo and Ma.

        https://ieeexplore.ieee.org/document/1710358

        We assume linear transition and measurement model in the
        following form
            x[k] = Fx[k-1] + w[k-1]
            z[k] = Hx[k] + v[k]
        Inputs:

        - model: dictionary which contains the following elements(keys are strings):

               F: state transition matrix

               H: measurement matrix

               Q: process noise covariance matrix(of variable w[k]).

               R: measurement noise covariance matrix(of variable v[k]).

             p_d: probability of target detection

             p_s: probability of target survival

            Spawning model, see pg. 5. of the paper. It's a Gaussian Mixture conditioned on state

             F_spawn:  d_spawn: Q_spawn: w_spawn: lists of ndarray objects with the same length, see pg. 5

            clutt_int_fun: reference to clutter intensity function, gets only one argument, which is the current measure

               T: U: Jmax: Pruning parameters, see pg. 7.

            birth_GM: The Gaussian Mixture of the birth intensity
        """
        self.Q = model["Q"]
        self.nObj = model["nObj"]
        self.birth_w = model["birth_w"]
        self.birth_P = model["birth_P"]
        # self.birth_GM = model["birth_GM"]
        self.specs = model["specs"]
        self.H = model["H"]
        self.R = model["R"]
        self.clutter_density_func = model["clutt_int_fun"]
        self.T = model["T"]
        self.U = model["U"]
        self.A = model["A"]
        self.alpha = model["alpha"]
        self.Jmax = model["Jmax"]

    def cosine_similarity(self, xc: torch.Tensor, yc: torch.Tensor):
        if yc.ndimension() == 1:
            dot_product = torch.dot(xc, yc)
            norm_x = torch.linalg.norm(xc)
            norm_y = torch.linalg.norm(yc)
        else:
            dot_product = torch.mv(yc, xc)
            norm_x = torch.linalg.norm(xc)
            norm_y = torch.linalg.norm(yc, dim=1)
            
        similarity = dot_product / (norm_x * norm_y + 1e-8)
        return torch.abs(similarity)

    def p_d_calc(self, distance: torch.Tensor, specs: torch.Tensor):
        max_range, threshold, constant_p_d, min_p_d = specs
        
        if distance <= threshold:
            return constant_p_d
        elif distance <= max_range:
            scale = (distance - threshold) / (max_range - threshold)
            return constant_p_d - scale * (constant_p_d - min_p_d)
        else:
            return min_p_d

    def generate_smoothed_cls(self, cls: torch.Tensor):
        total_count = torch.sum(cls)
        num_classes = cls.shape[0]
        smoothed_cls = (cls + self.alpha) / (total_count + self.alpha * num_classes)
        return smoothed_cls

    def prediction(self, v: VA_GaussianMixture) -> VA_GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # targets that survived v_s:
        v_s = spread_convariance(v, self.Q)
        # final phd of prediction
        return VA_GaussianMixture(v_s.w, v_s.m, v_s.P, v.cls)

    def correction(
        self, v: VA_GaussianMixture, p_v: torch.Tensor, Z: torch.Tensor, Zcls: torch.Tensor, distance: torch.Tensor
    ) -> VA_GaussianMixture:
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step
        - Z: Measurement set, containing set of observations
        """
        p_d = self.p_d_calc(distance, self.specs)
        
        # Residual calculations
        res_w = p_v * v.w * p_d
        res_m = v.m @ self.H.T
        res_P = self.H @ v.P @ self.H.T + self.R
        
        v_residual = VA_GaussianMixture(res_w, res_m, res_P, v.cls)
        detP = torch.linalg.det(v_residual.P)
        invP = torch.linalg.inv(v_residual.P)
        v_residual.set_covariance_determinant_and_inverse_list(detP, invP)

        K = v.P @ self.H.T @ invP
        P_kk = v.P - K @ self.H @ v.P

        # Missed detections
        w = [v.w * p_v * (1 - p_d), v.w * (1 - p_v)]
        m = [v.m, v.m]
        P = [v.P, v.P]
        cls = [v.cls, v.cls]

        for j in range(len(Z)):
            z = Z[j]
            values = v_residual.mixture_component_values_list(z)
            # Ensure clutter density function returns a torch tensor
            clutter = clutter_intensity_function(z, self.lc, self.surveillance_region)
            normalization_factor = torch.sum(values) + clutter
            
            obs_cls = torch.zeros(self.nObj, device=z.device)
            obs_cls[Zcls[j]] = 1.0
            obs_cls = self.generate_smoothed_cls(obs_cls)
            
            p_c = self.cosine_similarity(obs_cls, v_residual.cls) 
            w.append(p_c * values / normalization_factor)
            
            innov = (z - res_m).unsqueeze(-1)
            m.append(v.m + (K @ innov).squeeze(-1))
            P.append(P_kk)
            
            new_cls_unnorm = v_residual.cls + obs_cls
            cls.append(new_cls_unnorm / torch.sum(new_cls_unnorm, dim=1, keepdim=True))

            if torch.max(values) < self.A:
                w.append(self.birth_w.unsqueeze(0))
                m.append(z.unsqueeze(0))
                P.append(self.birth_P.unsqueeze(0))
                cls.append(obs_cls.unsqueeze(0))

        return VA_GaussianMixture(torch.cat(w), torch.cat(m), torch.cat(P), torch.cat(cls))
    def pruning(self, v: VA_GaussianMixture) -> VA_GaussianMixture:
        """
        See https://ieeexplore.ieee.org/document/7202905 for details
        """
        indices = (v.w > self.T).nonzero(as_tuple=True)[0]
        vw = v.w[indices]
        vm = v.m[indices]
        vP = v.P[indices]
        vcls = v.cls[indices]
        
        invP_2x2 = torch.linalg.inv(vP[:, :2, :2])
        
        w_final = []
        m_final = []
        P_final = []
        cls_final = []

        I = torch.arange(len(vw), device=vw.device).tolist()

        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            
            L = []
            for i in I:
                p_c = self.cosine_similarity(vcls[i], vcls[j])
                diff_2d = vm[i][:2] - vm[j][:2]
                dist = diff_2d @ invP_2x2[i] @ diff_2d
                
                if dist <= (self.U * p_c):
                    L.append(i)

            L_tensor = torch.tensor(L, device=vw.device)
            w_L = vw[L_tensor]
            m_L = vm[L_tensor]
            P_L = vP[L_tensor]
            cls_L = vcls[L_tensor]

            w_new = torch.sum(w_L)
            m_first_two = torch.sum((w_L.unsqueeze(1) * m_L[:, :2]), dim=0) / w_new
            m_last = torch.max(m_L[:, 2], dim=0)[0].unsqueeze(0)
            m_new = torch.cat([m_first_two, m_last])

            diff_m = m_new - m_L
            outer_p = torch.matmul(diff_m.unsqueeze(2), diff_m.unsqueeze(1))
            P_new = torch.sum(w_L.unsqueeze(1).unsqueeze(2) * (P_L + outer_p), dim=0) / w_new

            cls_new = torch.sum(w_L.unsqueeze(1) * cls_L, dim=0) / w_new

            w_final.append(w_new)
            m_final.append(m_new)
            P_final.append(P_new)
            cls_final.append(cls_new)
            
            I = [i for i in I if i not in L]

        w_out = torch.stack(w_final)
        m_out = torch.stack(m_final)
        P_out = torch.stack(P_final)
        cls_out = torch.stack(cls_final)

        if len(w_out) > self.Jmax:
            _, top_indices = torch.topk(w_out, self.Jmax)
            w_out = w_out[top_indices]
            m_out = m_out[top_indices]
            P_out = P_out[top_indices]
            cls_out = cls_out[top_indices]

        return VA_GaussianMixture(w_out, m_out, P_out, cls_out)

    def state_estimation(self, v: VA_GaussianMixture):
        X = []
        Xc = []
        
        indices = (v.w >= 0.5).nonzero(as_tuple=True)[0]
        
        if indices.numel() > 0:
            estimates_m = v.m[indices]
            estimates_cls = v.cls[indices]
            
            for i in range(len(indices)):
                X.append(estimates_m[i])
                max_val, max_idx = torch.max(estimates_cls[i], dim=0)
                Xc.append(max_idx.item())
                
        return X, Xc