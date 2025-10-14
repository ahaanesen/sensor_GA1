from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import scipy.linalg
from senfuslib import MultiVarGauss, DynamicModel
from states import (ErrorState, ImuMeasurement,
                    CorrectedImuMeasurement, NominalState,
                    GnssMeasurement, EskfState)
from quaternion import RotationQuaterion
from utils.indexing import block_3x3
from utils.cross_matrix import get_cross_matrix
from solution import models as models_solu


@dataclass
class ModelIMU:
    """The IMU is considered a dynamic model instead of a sensar. 
    This works as an IMU measures the change between two states, 
    and not the state itself.."""

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    accm_correction: 'np.ndarray[3, 3]'
    gyro_correction: 'np.ndarray[3, 3]'

    g: 'np.ndarray[3]' = field(default=np.array([0, 0, 9.82]))

    Q_c: 'np.ndarray[12, 12]' = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x): return np.diag([x]*3)

        accm_corr = self.accm_correction
        gyro_corr = self.gyro_correction

        self.Q_c = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accm_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.accm_bias_std**2),
            diag3(self.gyro_bias_std**2)
        )

    def correct_z_imu(self,
                      x_est_nom: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_est_nom: previous nominal state
            z_imu: raw IMU measurement

        Returns:
            z_corr: corrected IMU measurement
        """
        acc_est = self.accm_correction @ (z_imu.acc - x_est_nom.accm_bias)  # np.zeros(3) # (2) in oppgavetekst
        avel_est = self.gyro_correction @ (z_imu.avel - x_est_nom.gyro_bias) # np.zeros(3) # (4) in oppgavetekst

        z_corr = CorrectedImuMeasurement(acc_est, avel_est)
        # TODO remove this
        # z_corr = models_solu.ModelIMU.correct_z_imu(self, x_est_nom, z_imu)
        return z_corr

    def predict_nom(self,
                    x_est_nom: NominalState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement and a 
        time step, by discretizing (10.58) in the book.

        We assume the change in orientation is negligable when caculating 
        predicted position and velicity, see assignment pdf.

        Hint: You can use: delta_rot = RotationQuaterion.from_avec(something)

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_nom_pred: predicted nominal state
        """
        Rq = x_est_nom.ori.as_rotmat()
        acc_world = Rq @ z_corr.acc + self.g
        
        pos_pred = x_est_nom.pos + dt*x_est_nom.vel + (dt**2)/2*(acc_world) # np.zeros(3)  # TODO
        vel_pred = x_est_nom.vel + dt*(acc_world) # np.zeros(3)  # TODO

        delta_rot = RotationQuaterion.from_avec(z_corr.avel*dt)  # TODO
        ori_pred = x_est_nom.ori @ delta_rot  # np.zeros(3)  # TODO

        acc_bias_pred = x_est_nom.accm_bias# + dt*x_est_nom.accm_bias  # np.zeros(3)  # TODO
        gyro_bias_pred = x_est_nom.gyro_bias# + dt*x_est_nom.gyro_bias  # np.zeros(3)  # TODO

        # TODO remove this
        # x_nom_pred_1 = models_solu.ModelIMU.predict_nom(
        #     self, x_est_nom, z_corr, dt)
        x_nom_pred = NominalState(pos_pred, vel_pred, ori_pred, acc_bias_pred, gyro_bias_pred)

        return x_nom_pred

    def A_c(self,
            x_est_nom: NominalState,
            z_corr: CorrectedImuMeasurement,
            ) -> 'np.ndarray[15, 15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        ex: first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev: previous nominal state
            z_corr: corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A_c = np.zeros((15, 15))
        Rq = x_est_nom.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)

        A_c[block_3x3(0,1)] = np.eye(3)
        A_c[block_3x3(1,2)] = -Rq @ S_acc
        A_c[block_3x3(1,3)] = -Rq @ self.accm_correction
        A_c[block_3x3(2,2)] = -S_omega
        A_c[block_3x3(2,4)] = -self.gyro_correction
        # TODO remove this
        # A_c = models_solu.ModelIMU.A_c(self, x_est_nom, z_corr)
        return A_c

    def get_error_G_c(self,
                      x_est_nom: NominalState,
                      ) -> 'np.ndarray[15, 15]':
        """The continous noise covariance matrix, G, in (10.68)

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_est_nom: previous nominal state
        Returns:
            G_c (ndarray[15, 15]): G in (10.68)
        """
        G_c = np.zeros((15, 12))
        Rq = x_est_nom.ori.as_rotmat()
        
        G_c[block_3x3(1, 0)] = -Rq
        G_c[block_3x3(2, 1)] = -np.eye(3)
        G_c[block_3x3(3, 2)] = np.eye(3)
        G_c[block_3x3(4, 3)] = np.eye(3)

        # TODO remove this
        # G_c = models_solu.ModelIMU.get_error_G_c(self, x_est_nom)

        return G_c

    def get_discrete_error_diff(self,
                                x_est_nom: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                dt: float
                                ) -> Tuple['np.ndarray[15, 15]',
                                           'np.ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: Use scipy.linalg.expm to get the matrix exponential

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measurement
            dt: time step
        Returns:
            A_d (ndarray[15, 15]): discrede transition matrix
            GQGT_d (ndarray[15, 15]): discrete noise covariance matrix
        """
        A_c = self.A_c(x_est_nom, z_corr)  # TODO
        G_c = self.get_error_G_c(x_est_nom)  # TODO
        Q_c = self.Q_c
        GQGT_c = G_c @ Q_c @ G_c.T  # TODO

        n = A_c.shape[0]
        exponent = np.zeros((2*n, 2*n))  # TODO
        exponent[:n, :n] = -A_c
        exponent[:n, n:] = GQGT_c
        exponent[n:, n:] = A_c.T
        exponent = exponent * dt

        VanLoanMatrix = scipy.linalg.expm(exponent)  # TODO
        V1 = VanLoanMatrix[n:, n:]
        V2 = VanLoanMatrix[:n, n:]
        A_d = V1.T  # TODO
        GQGT_d = V1.T @ V2  # TODO

        # TODO remove this
        # A_d, GQGT_d = models_solu.ModelIMU.get_discrete_error_diff(
        #     self, x_est_nom, z_corr, dt)

        return A_d, GQGT_d

    def predict_err(self,
                    x_est_prev: EskfState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float,
                    ) -> MultiVarGauss[ErrorState]:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_est_prev: previous estimated eskf state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_err_pred: predicted error state gaussian
        """
        x_est_prev_nom = x_est_prev.nom
        x_est_prev_err = x_est_prev.err
        Ad, GQGTd = self.get_discrete_error_diff(x_est_prev_nom, z_corr, dt)  # TODO
        
        P_prev = x_est_prev_err.cov
        P_pred = Ad @ P_prev @ Ad.T + GQGTd  # TODO
        # P_pred = np.eye(15)  # TODO

        mean_pred = np.zeros(15)
        x_err_pred = MultiVarGauss[ErrorState](ErrorState.from_array(mean_pred), P_pred)

        # TODO remove this
        # x_err_pred = models_solu.ModelIMU.predict_err(
        #     self, x_est_prev, z_corr, dt)
        return x_err_pred
