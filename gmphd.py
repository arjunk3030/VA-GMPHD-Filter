import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any


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
    def __init__(self, w: List[np.float64], m: List[np.ndarray], P: List[np.ndarray]):
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

    def mixture_component_values_list(self, x: np.ndarray) -> List[float]:
        """
        Sometimes it is useful to have value of each component multiplied with its weight
        :param x: vector
        :return: List[np.float64]:
        List of components values at x, multiplied with their weight.
        """
        val = []
        if self.detP is None:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i]))
        else:
            for i in range(len(self.w)):
                val.append(
                    self.w[i]
                    * multivariate_gaussian_predefined_det_and_inv(
                        x, self.m[i], self.detP[i], self.invP[i]
                    )
                )
        return val

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        return GaussianMixture(w, m, P)


def get_matrices_inverses(P_list: List[np.ndarray]) -> List[np.ndarray]:
    inverse_P_list = []
    for P in P_list:
        inverse_P_list.append(lin.inv(P))
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
    return GaussianMixture(w, m, P)


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
        # to do: dtype, copy, improve performance
        self.p_s = model["p_s"]
        self.F = model["F"]
        self.Q = model["Q"]
        self.w_spawn = model["w_spawn"]
        self.F_spawn = model["F_spawn"]
        self.d_spawn = model["d_spawn"]
        self.Q_spawn = model["Q_spawn"]
        self.birth_GM = model["birth_GM"]
        self.p_d = model["p_d"]
        self.H = model["H"]
        self.R = model["R"]
        self.clutter_density_func = model["clutt_int_fun"]
        self.T = model["T"]
        self.U = model["U"]
        self.Jmax = model["Jmax"]

    def spawn_mixture(self, v: GaussianMixture) -> GaussianMixture:
        """
        Spawning targets in prediction step
        """
        w = []
        m = []
        P = []
        for i, w_v in enumerate(v.w):
            for j, w_spawn in enumerate(self.w_spawn):
                w.append(w_v * w_spawn)
                m.append(self.F_spawn[j] @ v.m[i] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ v.P[i] @ self.F_spawn[j].T)
        return GaussianMixture(w, m, P)

    def prediction(self, v: GaussianMixture) -> GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # v_pred = v_s + v_spawn +  v_new_born
        birth_copy = self.birth_GM.copy()
        # targets that survived v_s:
        v_s = thinning_and_displacement(v, self.p_s, self.F, self.Q)
        # spawning targets
        v_spawn = self.spawn_mixture(v)
        # final phd of prediction
        return GaussianMixture(
            v_s.w + v_spawn.w + birth_copy.w,
            v_s.m + v_spawn.m + birth_copy.m,
            v_s.P + v_spawn.P + birth_copy.P,
        )

    def correction(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step
        - Z: Measurement set, containing set of observations
        """
        v_residual = thinning_and_displacement(v, self.p_d, self.H, self.R)
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
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P

        for z in Z:
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(P_kk[i].copy())
        print(
            f"z {len(Z)}, v_copy {len(v_copy.w)}, v_residiual {len(v_residual.w)}  final {len(w)}"
        )
        print(w)
        return GaussianMixture(w, m, P)

    def pruning(self, v: GaussianMixture) -> GaussianMixture:
        """
        See https://ieeexplore.ieee.org/document/7202905 for details
        """
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        v = GaussianMixture(w, m, P)
        I = (np.array(v.w) > self.T).nonzero()[0].tolist()
        invP = get_matrices_inverses(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        w = []
        m = []
        P = []
        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            L = []
            for i in I:
                x = (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j])
                if (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j]) <= self.U:
                    L.append(i)
            w_new = np.sum(vw[L])
            m_new = np.sum((vw[L] * vm[L].T).T, axis=0) / w_new
            P_new = np.zeros((m_new.shape[0], m_new.shape[0]))
            for i in L:
                P_new += vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))
            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            I = [i for i in I if i not in L]

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax :]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]

        return GaussianMixture(w, m, P)

    def state_estimation(self, v: GaussianMixture) -> List[np.ndarray]:
        X = []
        for i in range(len(v.w)):
            if v.w[i] >= 0.5:
                # for j in range(int(np.round(v.w[i]))):
                X.append(v.m[i])
        return X

    def filter_data(self, Z: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Given the list of collections of measurements for each time step, perform filtering and return the
        estimated sets of tracks for each step.

        :param Z: list of observations(measurements) for each time step
        :return X:
        list of estimated track sets for each time step
        """
        X = []
        v = GaussianMixture([], [], [])
        for z in Z:
            v = self.prediction(v)
            v = self.correction(v, z)
            v = self.pruning(v)
            x = self.state_estimation(v)
            X.append(x)
        return X
