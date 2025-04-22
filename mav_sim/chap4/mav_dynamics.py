"""
mavDynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

part of mavPySim
    - Beard & McLain, PUP, 2012
    - Update history:
        12/20/2018 - RWB
"""
from typing import Optional, cast

import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np

# load mav dynamics from previous chapter
from mav_sim.chap3.mav_dynamics import IND, DynamicState, ForceMoments, derivatives
from mav_sim.message_types.msg_delta import MsgDelta

# load message types
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools import types
from mav_sim.tools.rotations import Quaternion2Euler, Quaternion2Rotation


class MavDynamics:
    """Implements the dynamics of the MAV using vehicle inputs and wind
    """

    def __init__(self, Ts: float, state: Optional[DynamicState] = None):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        if state is None:
            self._state = DynamicState().convert_to_numpy()
        else:
            self._state = state.convert_to_numpy()

        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec

        # update velocity data
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state)

        # Update forces and moments data
        self._forces = np.array([[0.], [0.], [0.]]) # store forces to avoid recalculation in the sensors function (ch 7)
        self._moments = np.array([[0.], [0.], [0.]]) # store moments to avoid recalculation
        forces_moments_vec = forces_moments(self._state, MsgDelta(), self._Va, self._beta, self._alpha)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)


        # initialize true_state message
        self.true_state = MsgState()
        self._update_true_state()

    ###################################
    # public functions
    def update(self, delta: MsgDelta, wind: types.WindVector, time_step: Optional[float] = None) -> None:
        """
        Integrate the differential equations defining dynamics, update sensors

        Args:
            delta : (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind: the wind vector in inertial coordinates
        """
        # get forces and moments acting on rigid bod
        forces_moments_vec = forces_moments(self._state, delta, self._Va, self._beta, self._alpha)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)

        # Get the timestep
        if time_step is None:
            time_step = self._ts_simulation

        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = derivatives(self._state, forces_moments_vec)
        k2 = derivatives(self._state + time_step/2.*k1, forces_moments_vec)
        k3 = derivatives(self._state + time_step/2.*k2, forces_moments_vec)
        k4 = derivatives(self._state + time_step*k3, forces_moments_vec)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(IND.E0)
        e1 = self._state.item(IND.E1)
        e2 = self._state.item(IND.E2)
        e3 = self._state.item(IND.E3)
        norm_e = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[IND.E0][0] = self._state.item(IND.E0)/norm_e
        self._state[IND.E1][0] = self._state.item(IND.E1)/norm_e
        self._state[IND.E2][0] = self._state.item(IND.E2)/norm_e
        self._state[IND.E3][0] = self._state.item(IND.E3)/norm_e

        # update the airspeed, angle of attack, and side slip angles using new state
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state, wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state: types.DynamicState) -> None:
        """Loads a new state
        """
        self._state = new_state

    def get_state(self) -> types.DynamicState:
        """Returns the state
        """
        return self._state

    def get_struct_state(self) ->DynamicState:
        '''Returns the current state in a struct format

        Outputs:
            DynamicState: The latest state of the mav
        '''
        return DynamicState(self._state)

    def get_fm_struct(self) -> ForceMoments:
        '''Returns the latest forces and moments calculated in dynamic update'''
        force_moment = np.zeros((6,1))
        force_moment[0:3] = self._forces
        force_moment[3:6] = self._moments
        return ForceMoments(force_moment= cast(types.ForceMoment, force_moment) )

    def get_euler(self) -> tuple[float, float, float]:
        '''Returns the roll, pitch, and yaw Euler angles based upon the state'''
        # Get Euler angles
        phi, theta, psi = Quaternion2Euler(self._state[IND.QUAT])

        # Return angles
        return (phi, theta, psi)

    ###################################
    # private functions
    def _update_true_state(self) -> None:
        """ update the class structure for the true state:

        [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        """
        quat = cast(types.Quaternion, self._state[IND.QUAT])
        phi, theta, psi = Quaternion2Euler(quat)
        pdot = Quaternion2Rotation(quat) @ self._state[IND.VEL]
        self.true_state.north = self._state.item(IND.NORTH)
        self.true_state.east = self._state.item(IND.EAST)
        self.true_state.altitude = -self._state.item(IND.DOWN)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = cast(float, np.linalg.norm(pdot))
        if self.true_state.Vg != 0.:
            self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        else:
            self.true_state.gamma = 0.
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(IND.P)
        self.true_state.q = self._state.item(IND.Q)
        self.true_state.r = self._state.item(IND.R)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)

def forces_moments(state: types.DynamicState, delta: MsgDelta, Va: float, beta: float, alpha: float) -> types.ForceMoment:
    """
    Return the forces on the UAV based on the state, wind, and control surfaces

    Args:
        state: current state of the aircraft
        delta: flap and thrust commands
        Va: Airspeed
        beta: Side slip angle
        alpha: Angle of attack


    Returns:
        Forces and Moments on the UAV (in body frame) np.matrix(fx, fy, fz, Mx, My, Mz)
    """
    st = DynamicState(state)

    # get prop thrust/torque
    [Tp, Qp] = motor_thrust_torque(Va, delta.throttle)

    # compute gravitational force
    force_gravity = MAV.mass * MAV.gravity * np.array([[2 * (st.e1 * st.e3 - st.e2 * st.e0)], 
                                                        [2 * (st.e2 * st.e3 + st.e1 * st.e0)], 
                                                        [st.e3**2 + st.e0**2 - st.e1**2 - st.e2**2]])

    sigma = (1 + np.exp(-MAV.M * (alpha - MAV.alpha0)) + np.exp(MAV.M * (alpha + MAV.alpha0))) / \
        ((1 + np.exp(-MAV.M * (alpha - MAV.alpha0))) * (1 + np.exp(MAV.M * (alpha + MAV.alpha0))))
    C_L = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + sigma * (2 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha))
    C_D = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * alpha)**2 / (np.pi * MAV.e * MAV.AR)

    Cx = -C_D * np.cos(alpha) + C_L * np.sin(alpha)
    Cxq = -MAV.C_D_q * np.cos(alpha) + MAV.C_L_q * np.sin(alpha)
    Cx_delta_e = -MAV.C_D_delta_e * np.cos(alpha) + MAV.C_L_delta_e * np.sin(alpha)

    Cz = -C_D * np.sin(alpha) - C_L * np.cos(alpha)
    Czq = -MAV.C_D_q * np.sin(alpha) - MAV.C_L_q * np.cos(alpha)
    Cz_delta_e = -MAV.C_D_delta_e * np.sin(alpha) - MAV.C_L_delta_e * np.cos(alpha)

    bodyForces = force_gravity + np.array([[Tp], [0], [0]])

    if Va != 0:
        bodyForces = force_gravity + 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * np.array([
                    [Cx + Cxq * MAV.c / (2 * Va) * st.q], 
                    [MAV.C_Y_0 + MAV.C_Y_beta * beta + MAV.C_Y_p * MAV.b / \
                            (2 * Va) * st.p + MAV.C_Y_r * MAV.b  / (2 * Va) * st.r], 
                    [Cz + Czq * MAV.c / (2 * Va) * st.q]
                ]) + 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * np.array([
                    [Cx_delta_e * delta.elevator],
                    [MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder],
                    [Cz_delta_e * delta.elevator]
                ]) + np.array([[Tp], [0], [0]])

    # Extract elements
    fx = bodyForces[0][0]
    fy = bodyForces[1][0]
    fz = bodyForces[2][0]

    Mx = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * \
            (MAV.C_ell_0 + MAV.C_ell_beta * beta + MAV.C_ell_delta_a * delta.aileron + \
                MAV.C_ell_delta_r * delta.rudder) - Qp
    My = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.c * \
            (MAV.C_m_0 + MAV.C_m_alpha * alpha + MAV.C_m_delta_e * delta.elevator)
    Mz = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * \
            (MAV.C_n_0 + MAV.C_n_beta * beta + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)

    if Va != 0:
        Mx = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * \
            (MAV.C_ell_0 + MAV.C_ell_beta * beta + MAV.C_ell_p * MAV.b / (2 * Va) * st.p + \
                MAV.C_ell_r * MAV.b / (2 * Va) * st.r + \
                    MAV.C_ell_delta_a * delta.aileron + \
                        MAV.C_ell_delta_r * delta.rudder) - Qp
        My = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.c * \
            (MAV.C_m_0 + MAV.C_m_alpha * alpha + \
                MAV.C_m_q * MAV.c / (2 * Va) * st.q + \
                    MAV.C_m_delta_e * delta.elevator)
        Mz = 1 / 2 * MAV.rho * Va**2 * MAV.S_wing * MAV.b * \
            (MAV.C_n_0 + MAV.C_n_beta * beta + \
                MAV.C_n_p * MAV.b / (2 * Va) * st.p + \
                    MAV.C_n_r * st.r * MAV.b / (2 * Va) + \
                        MAV.C_n_delta_a * delta.aileron + \
                            MAV.C_n_delta_r * delta.rudder)

    return types.ForceMoment( np.array([[fx, fy, fz, Mx, My, Mz]]).T )

def motor_thrust_torque(Va: float, delta_t: float) -> tuple[float, float]:
    """ compute thrust and torque due to propeller  (See addendum by McLain)

    Args:
        Va: Airspeed
        delta_t: Throttle command

    Returns:
        T_p: Propeller thrust
        Q_p: Propeller torque
    """
    Vin = MAV.V_max * delta_t
    a = MAV.rho * MAV.D_prop**5 / (4 * np.pi**2) * MAV.C_Q0
    b = (MAV.rho * MAV.D_prop**4 / (2 * np.pi) * MAV.C_Q1 * Va) + (MAV.KQ * MAV.KQ / MAV.R_motor)
    c = (MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2) - (MAV.KQ / MAV.R_motor * Vin) + (MAV.KQ * MAV.i0)
    Omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # thrust and torque due to propeller
    thrust_prop = ((MAV.rho * MAV.D_prop**4 * MAV.C_T0) / \
        (4 * np.pi**2)) * Omega_p**2 + \
            ((MAV.rho * MAV.D_prop**3 * MAV.C_T1 * Va) / (2 * np.pi)) * Omega_p + \
                (MAV.rho * MAV.D_prop**2 * MAV.C_T2 * Va**2)
    torque_prop = (MAV.rho * MAV.D_prop**5 * MAV.C_Q0 / \
        (4 * np.pi**2)) * Omega_p**2 + \
            (MAV.rho * MAV.D_prop**4 * MAV.C_Q1 * Va / (2 * np.pi)) * Omega_p + \
                (MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2)


    return thrust_prop, torque_prop

def update_velocity_data(state: types.DynamicState, \
    wind: types.WindVector = types.WindVector( np.zeros((6,1)) ) 
    )  -> tuple[float, float, float, types.NP_MAT]:
    """Calculates airspeed, angle of attack, sideslip, and velocity wrt wind

    Args:
        state: current state of the aircraft

    Returns:
        Va: Airspeed
        alpha: Angle of attack
        beta: Side slip angle
        wind_inertial_frame: Wind vector in inertial frame
    """
    st = DynamicState(state)

    steady_state = wind[0:3]
    gust = wind[3:6]

    # convert wind vector from world to body frame
    R = Quaternion2Rotation(state[IND.QUAT]) # rotation from body to world frame
    wind_body_frame = R.T @ steady_state  # rotate steady state wind to body frame
    wind_body_frame += gust  # add the gust
    wind_inertial_frame = R @ wind_body_frame # Wind in the world frame

    ur = st.u - wind_body_frame[0][0]
    vr = st.v - wind_body_frame[1][0]
    wr = st.w - wind_body_frame[2][0]

    # compute airspeed
    Va = np.sqrt(ur**2 + vr**2 + wr**2)

    # compute angle of attack
    alpha = np.arctan2(wr, ur)

    # compute sideslip angle
    beta = 0
    if Va != 0:
        beta = np.arcsin(vr / Va)

    # Return computed values
    return (Va, alpha, beta, wind_inertial_frame)
