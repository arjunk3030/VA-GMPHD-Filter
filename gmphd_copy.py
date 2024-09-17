import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any
import Constants


def multivariate_gaussian(x: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """
    Multivatiate Gaussian Distribution

    :param x: vector
    :param m: distribution mean vector
    :param P: Covariance matrix
    :return: probability density function at x
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (lin.det(P) ** 0.5))
    second_part = -0.5 * (x - m) @ lin.inv(P) @ (x - m)
    return first_part * np.exp(second_part)


def multivariate_gaussian_predefined_det_and_inv(
    x: np.ndarray, m: np.ndarray, detP: np.float64, invP: np.ndarray
) -> float:
    """
    Multivariate Gaussian Distribution with provided determinant and inverse of the Gaussian mixture.
    Useful in case when we already have precalculted determinant and inverse of the covariance matrix.
    :param x: vector
    :param m: distribution mean
    :param detP: determinant of the covariance matrix
    :param invP: inverse of the covariance matrix
    :return: probability density function at x
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (detP**0.5))
    second_part = -0.5 * (x - m) @ invP @ (x - m)
    return first_part * np.exp(second_part)


def clutter_intensity_function(z: np.ndarray, lc: int, surveillance_region: np.ndarray):
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
        # example in two dimensions: lc/((xmax - xmin)*(ymax-ymin))
        return lc / (
            (surveillance_region[0][1] - surveillance_region[0][0])
            * (surveillance_region[1][1] - surveillance_region[1][0])
        )
    else:
        return 0.0


class GaussianMixture:
    def __init__(
        self,
        w: List[np.float64],
        m: List[np.ndarray],
        P: List[np.ndarray],
        cls: List[np.ndarray],
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
        self, detP: List[np.float64], invP: List[np.ndarray]
    ):
        """
        For each Gaussian component, provide the determinant and the covariance inverse
        :param detP: list of determinants for each Gaussian component in the mixture
        :param invP: list of covariance inverses for each Gaussian component in the mixture
        """
        self.detP = detP
        self.invP = invP

    def mixture_value(self, x: np.ndarray):
        """
        Gaussian Mixture function for the given vector x
        """
        sum = 0
        if self.detP is None:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian_predefined_det_and_inv(
                    x, self.m[i], self.detP[i], self.invP[i]
                )
        return sum

    def mixture_single_component_value(self, x: np.ndarray, i: int) -> float:
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

    # def mixture_component_values_list(self, x: np.ndarray) -> List[float]:
    #     """
    #     Sometimes it is useful to have value of each component multiplied with its weight
    #     :param x: vector
    #     :return: List[np.float64]:
    #     List of components values at x, multiplied with their weight.
    #     """
    #     val = []
    #     if self.detP is None:
    #         for i in range(len(self.w)):
    #             val.append(self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i]))
    #     else:
    #         for i in range(len(self.w)):
    #             val.append(
    #                 self.w[i]
    #                 * multivariate_gaussian_predefined_det_and_inv(
    #                     x, self.m[i], self.detP[i], self.invP[i]
    #                 )
    #             )
    #     return val

    def mixture_component_values_list(self, x: np.ndarray) -> List[float]:
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
        return val

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        cls = self.cls.copy()
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        return GaussianMixture(w, m, P, cls)


def get_matrices_inverses(P_list: List[np.ndarray]) -> List[np.ndarray]:
    inverse_P_list = []
    for P in P_list:
        inverse_P_list.append(np.linalg.inv(P))
    return inverse_P_list


def get_matrices_determinants(P_list: List[np.ndarray]) -> List[float]:
    """
    :param P_list: list of covariance matrices
    :return:
    """
    detP = []
    for P in P_list:
        detP.append(lin.det(P))
    return detP


def thinning_and_displacement(v: GaussianMixture, p, F: np.ndarray, Q: np.ndarray):
    """
    For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
    See https://ieeexplore.ieee.org/document/7202905 for details
    """
    w = []
    m = []
    P = []
    for weight in v.w:
        w.append(weight * p)
    for mean in v.m:
        m.append(F @ mean)
    for cov_matrix in v.P:
        P.append(Q + F @ cov_matrix @ F.T)
    return GaussianMixture(w, m, P, v.cls)


def just_displacement(v: GaussianMixture, F: np.ndarray, Q: np.ndarray):
    """
    For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
    See https://ieeexplore.ieee.org/document/7202905 for details
    """
    m = []
    P = []
    for mean in v.m:
        m.append(F @ mean)
    for cov_matrix in v.P:
        P.append(Q + F @ cov_matrix @ F.T)
    return GaussianMixture(v.w, m, P, v.cls)


def spread_convariance(v: GaussianMixture, Q: np.ndarray):
    w = []
    m = []
    P = []

    for weight in v.w:
        w.append(weight)

    for mean in v.m:
        m.append(mean)

    for cov_matrix in v.P:
        P.append(Q + cov_matrix)

    return GaussianMixture(w, m, P, v.cls)


class GmphdFilter:
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

    def cosine_similarity(self, xc, yc):
        xc = np.array(xc)
        yc = np.array(yc)

        dot_product = np.dot(xc, yc)
        magnitudex = np.linalg.norm(xc)
        magnitudey = np.linalg.norm(yc)

        similarity = dot_product / (magnitudex * magnitudey)

        return np.abs(similarity)

    def p_d_calc(self, distance: float, specs):
        max_range, threshold, constant_p_d, min_p_d = specs
        if distance <= threshold:
            return constant_p_d
        elif threshold < distance <= max_range:
            scale = (distance - threshold) / (max_range - threshold)
            return constant_p_d - scale * (constant_p_d - min_p_d)
        else:
            return min_p_d

    def generate_smoothed_cls(self, cls):
        total_count = sum(cls)
        smoothed_cls = [
            (count + self.alpha) / (total_count + self.alpha * len(cls))
            for count in cls
        ]
        return smoothed_cls

    def prediction(self, v: GaussianMixture) -> GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # targets that survived v_s:
        v_s = spread_convariance(v, self.Q)
        # final phd of prediction
        return GaussianMixture(v_s.w, v_s.m, v_s.P, v.cls)

    def correction(
        self, v: GaussianMixture, p_v, Z: List[np.ndarray], Zcls, distance
    ) -> GaussianMixture:
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step
        - Z: Measurement set, containing set of observations
        """
        p_d = self.p_d_calc(distance, self.specs)
        print(f"p_d is {p_d}")
        v_residual = GaussianMixture([], [], [], v.cls)
        for i, (w, m, P) in enumerate(zip(v.w, v.m, v.P)):
            v_residual.w.append(p_v[i] * w * p_d)
            v_residual.m.append(self.H @ m)
            v_residual.P.append(self.R + self.H @ P @ self.H.T)

        detP = get_matrices_determinants(v_residual.P)
        invP = get_matrices_inverses(v_residual.P)
        v_residual.set_covariance_determinant_and_inverse_list(detP, invP)

        K = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        v_copy = v.copy()

        w_visible = [
            weight * probability * (1 - p_d)
            for weight, probability in zip(v_copy.w, p_v)
        ]
        w_not_visible = [
            weight * (1 - probability) for weight, probability in zip(v_copy.w, p_v)
        ]
        w = w_visible + w_not_visible
        m = v_copy.m + v_copy.m
        P = v_copy.P + v_copy.P
        cls = v_copy.cls + v_copy.cls

        for z, z_cls in zip(Z, Zcls):
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            observation_cls = [0] * self.nObj
            observation_cls[z_cls] += 1
            observation_cls = self.generate_smoothed_cls(observation_cls)
            m_weight = 0
            for i in range(len(v_residual.w)):
                if values[i] > m_weight:
                    m_weight = values[i]
                p_c = self.cosine_similarity(observation_cls, v_residual.cls[i])
                w.append(p_c * values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(P_kk[i].copy())
                new_cls = [
                    current + obs
                    for current, obs in zip(v_residual.cls[i], observation_cls)
                ]
                total = sum(new_cls)
                new_cls = [x / total for x in new_cls]

                cls.append(new_cls)
            if m_weight < self.A:
                w.append(self.birth_w)
                m.append(z)
                P.append(self.birth_P)
                cls.append(observation_cls)
        return GaussianMixture(w, m, P, cls)

    def pruning(self, v: GaussianMixture) -> GaussianMixture:
        """
        See https://ieeexplore.ieee.org/document/7202905 for details
        """
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        cls = [v.cls[i] for i in I]
        v = GaussianMixture(w, m, P, cls)

        I = (np.array(v.w) > self.T).nonzero()[0].tolist()
        invP = get_matrices_inverses(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        vcls = np.array(v.cls)
        w = []
        m = []
        P = []
        cls = []

        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            L = []
            # for i in I:
            #     p_c = self.cosine_similarity(vcls[i], vcls[j])
            #     x = (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j])
            #     adjusted_x = x * 1 / p_c

            #     # Apply the merging condition
            #     if adjusted_x <= self.U:
            #         L.append(i)

            for i in I:
                p_c = self.cosine_similarity(vcls[i], vcls[j])

                if (vm[i][:2] - vm[j][:2]) @ invP[i][:2, :2] @ (
                    vm[i][:2] - vm[j][:2]
                ) <= (self.U * p_c):
                    L.append(i)

                # if (vm[i][:3] - vm[j][:3]) @ invP[i][:3, :3] @ (
                #     vm[i][:3] - vm[j][:3]
                # ) <= (self.U * p_c):
                #     L.append(i)

            w_new = np.sum(vw[L])
            # m_new = np.sum((vw[L] * vm[L].T).T, axis=0) / w_new
            m_first_two = np.sum((vw[L] * vm[L, :2].T).T, axis=0) / w_new
            m_last = vm[L, 2].max()
            m_new = np.concatenate([m_first_two, [m_last]])

            P_new = np.zeros((m_new.shape[0], m_new.shape[0]))
            cls_weighted_sum = np.zeros(len(vcls[L][0]))
            for i in L:
                cls_weighted_sum += vw[i] * np.array(vcls[i])
            w_new = np.sum(vw[L])
            cls_new = (cls_weighted_sum / w_new).tolist()
            for i in L:
                P_new += vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))

            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            cls.append(cls_new)
            I = [i for i in I if i not in L]

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax :]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]
            cls = [cls[i] for i in L]

        return GaussianMixture(w, m, P, cls)

    def state_estimation(self, v: GaussianMixture) -> List[np.ndarray]:
        X = []
        Xc = []
        for i in range(len(v.w)):
            if v.w[i] >= 0.5:
                max_value = max(v.cls[i])
                X.append(v.m[i])
                Xc.append(
                    v.cls[i].index(max_value),
                )
        return X, Xc
