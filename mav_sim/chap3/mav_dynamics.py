"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:
        12/17/2018 - RWB
        1/14/2019 - RWB
        12/21 - GND
"""
from typing import Optional, cast

import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np

# load message types
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools import types
from mav_sim.tools.rotations import (
    Euler2Quaternion,
    Quaternion2Euler,
    Quaternion2Rotation,
)


# Indexing constants for state
class StateIndices:
    """Constant class for easy access of state indices
    """
    NORTH: int  = 0  # North position
    EAST: int   = 1  # East position
    DOWN: int   = 2  # Down position
    U: int      = 3  # body-x velocity
    V: int      = 4  # body-y velocity
    W: int      = 5  # body-z velocity
    E0: int     = 6  # Quaternion - element 0
    E1: int     = 7  # Quaternion - element 1
    E2: int     = 8  # Quaternion - element 2
    E3: int     = 9  # Quaternion - element 3
    P: int      = 10 # roll rate - body frame - i
    Q: int      = 11 # pitch rate - body frame - j
    R: int      = 12 # yaw rate - body frame - k
    QUAT: list[int] = [E0, E1, E2, E3] # Quaternion indices
    VEL: list[int] = [U, V, W] # Body velocity indices
    NUM_STATES: int = 13 # Number of states
IND = StateIndices()

class DynamicState:
    """Struct for the dynamic state
    """
    def __init__(self, state: Optional[types.DynamicState] = None ) -> None:
        self.north: float     # North position
        self.east: float      # East position
        self.down: float      # Down position
        self.u: float         # body-x velocity
        self.v: float         # body-y velocity
        self.w: float         # body-z velocity
        self.e0: float        # quaternion - element 0
        self.e1: float        # quaternion - element 1
        self.e2: float        # quaternion - element 2
        self.e3: float        # quaternion - element 3
        self.p: float         # roll rate - body frame - i
        self.q: float         # pitch rate - body frame - j
        self.r: float         # yaw rate - body frame - k

        self.extract_state(state)

    def extract_state(self, state: Optional[types.DynamicState] = None ) ->None:
        """Initializes the state variables

        If no state is given, the defaults will be taken from mav_sim.parameters.aerosonde_parameters

        Args:
            state: Optional state from which to extract the state values

        """
        if state is None:
            self.north = MAV.north0 # North position
            self.east = MAV.east0   # East position
            self.down = MAV.down0   # Down position
            self.u = MAV.u0         # body-x velocity
            self.v = MAV.v0         # body-y velocity
            self.w = MAV.w0         # body-z velocity
            self.e0 = MAV.e0        # quaternion - element 0
            self.e1 = MAV.e1        # quaternion - element 1
            self.e2 = MAV.e2        # quaternion - element 2
            self.e3 = MAV.e3        # quaternion - element 3
            self.p = MAV.p0         # roll rate - body frame - i
            self.q = MAV.q0         # pitch rate - body frame - j
            self.r = MAV.r0         # yaw rate - body frame - k
        else:
            self.north = state.item(IND.NORTH)
            self.east =  state.item(IND.EAST)
            self.down =  state.item(IND.DOWN)
            self.u =     state.item(IND.U)
            self.v =     state.item(IND.V)
            self.w =     state.item(IND.W)
            self.e0 =    state.item(IND.E0)
            self.e1 =    state.item(IND.E1)
            self.e2 =    state.item(IND.E2)
            self.e3 =    state.item(IND.E3)
            self.p =     state.item(IND.P)
            self.q =     state.item(IND.Q)
            self.r =     state.item(IND.R)

    def convert_to_numpy(self) -> types.DynamicState:
        """Converts the state to a numpy object
        """
        output = np.empty( (IND.NUM_STATES,1) )
        output[IND.NORTH, 0] = self.north
        output[IND.EAST, 0] = self.east
        output[IND.DOWN, 0] = self.down
        output[IND.U, 0] = self.u
        output[IND.V, 0] = self.v
        output[IND.W, 0] = self.w
        output[IND.E0, 0] = self.e0
        output[IND.E1, 0] = self.e1
        output[IND.E2, 0] = self.e2
        output[IND.E3, 0] = self.e3
        output[IND.P, 0] = self.p
        output[IND.Q, 0] = self.q
        output[IND.R, 0] = self.r

        return output

    def set_attitude_euler(self, phi: float, theta: float, psi: float) -> None:
        """Sets the attitude of the state using euler coordinates

        Args:
            phi: roll (rotation about i^{v2})
            theta: pitch (rotation about j^{v1})
            psi: yaw (rotation about k^v)
        """
        # Calculate the quaternion
        e = Euler2Quaternion(phi, theta, psi)

        # Set the quaternion
        self.e0 = e.item(0)
        self.e1 = e.item(1)
        self.e2 = e.item(2)
        self.e3 = e.item(3)

    def extract_euler(self) -> tuple[float, float, float]:
        """Calculates the euler coordinates from the quaterion values

        Returns:
            (phi, theta, psi): Tuple of euler angles
        """

        # Extract the quaternion
        quat = cast(types.Quaternion, np.array([[self.e0],[self.e1], [self.e2], [self.e3] ]) )

        # Calculate the Euler coordinates
        return Quaternion2Euler(quat)

class ForceMoments:
    """Struct for forces and moments
    """
    IND_FX: int = 0     # Force in x-body direction
    IND_FY: int = 1     # Force in y-body direction
    IND_FZ: int = 2     # Force in z-body direction
    IND_L: int = 3      # Moment about x-body axis
    IND_M: int = 4      # Moment about y-body axis
    IND_N: int = 5      # Moment about z-body axis

    def __init__(self, force_moment: types.ForceMoment = cast(types.ForceMoment, np.zeros([6,1]))) -> None:
        self.fx = force_moment.item(self.IND_FX)
        self.fy = force_moment.item(self.IND_FY)
        self.fz = force_moment.item(self.IND_FZ)
        self.l = force_moment.item(self.IND_L)
        self.m = force_moment.item(self.IND_M)
        self.n = force_moment.item(self.IND_N)

    def to_array(self) -> types.ForceMoment:
        """ Convert the struct to an array of values

        Returns:
            force_moment: 6x1 array of forces and moments
        """
        force_moment = np.zeros([6,1])
        force_moment[self.IND_FX,0] = self.fx
        force_moment[self.IND_FY,0] = self.fy
        force_moment[self.IND_FZ,0] = self.fz
        force_moment[self.IND_L,0] = self.l
        force_moment[self.IND_M,0] = self.m
        force_moment[self.IND_N,0] = self.n

        return cast(types.ForceMoment, force_moment)



class MavDynamics:
    """Implements the dynamics of the MAV assuming forces and moments are directly input
    """

    def __init__(self, Ts: float, state: Optional[DynamicState] = None) -> None:
        """Initialize the dynamic variables

        Args:
            Ts: Time step in the simulation between function calls
        """
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        if state is None:
            self._state = DynamicState().convert_to_numpy()
        else:
            self._state = state.convert_to_numpy()
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments: types.ForceMoment, time_step: Optional[float] = None) -> None:
        '''Update states.

        Integrate the differential equations defining dynamics.
        Inputs are the forces and moments on the aircraft.

        Args:
            forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

        '''

        # Get the timestep
        if time_step is None:
            time_step = self.ts_simulation

        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = derivatives(self._state, forces_moments)
        k2 = derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = derivatives(self._state + time_step*k3, forces_moments)
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

        # update the message class for the true state
        self._update_true_state()

    def get_state(self) ->DynamicState:
        '''Returns the current state in a struct format

        Outputs:
            DynamicState: The latest state of the mav
        '''
        return DynamicState(self._state)

    def get_euler(self) -> tuple[float, float, float]:
        '''Returns the roll, pitch, and yaw Euler angles based upon the state'''
        # Get Euler angles
        phi, theta, psi = Quaternion2Euler(self._state[IND.QUAT])

        # Return angles
        return (phi, theta, psi)

    ###################################
    # private functions
    def _update_true_state(self) -> None:
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[IND.QUAT])
        self.true_state.north = self._state.item(IND.NORTH)
        self.true_state.east = self._state.item(IND.EAST)
        self.true_state.altitude = -self._state.item(IND.DOWN)
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.p = self._state.item(IND.P)
        self.true_state.q = self._state.item(IND.Q)
        self.true_state.r = self._state.item(IND.R)

def derivatives(state: types.DynamicState, forces_moments: types.ForceMoment) -> types.DynamicState:
    """Implements the dynamics xdot = f(x, u) where u is the force/moment vector

    Args:
        state: Current state of the vehicle
        forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

    Returns:
        Time derivative of the state ( f(x,u), where u is the force/moment vector )
    """
    st = DynamicState(state)
    
    fm = ForceMoments(forces_moments)

    rot_e_b_i = Quaternion2Rotation(state[IND.QUAT])

    pned_dot = rot_e_b_i @ np.array([[st.u], [st.v], [st.w]])
    
    uvw_dot = np.array([[st.r * st.v - st.q * st.w], 
                        [st.p * st.w - st.r * st.u],
                        [st.q * st.u - st.p * st.v]]) + (1 / MAV.mass) * np.array([[fm.fx], [fm.fy], [fm.fz]])
    
    e_dot = 1/2 * np.array([[0, -st.p, -st.q, -st.r],
                            [st.p, 0, st.r, -st.q],
                            [st.q, -st.r, 0, st.p],
                            [st.r, st.q, -st.p, 0]]) @ np.array([st.e0, st.e1, st.e2, st.e3]).transpose()

    pqrt_dot = np.array([[MAV.gamma1 * st.p * st.q - MAV.gamma2 * st.q * st.r],
                        [MAV.gamma5 * st.p * st.r - MAV.gamma6 * (st.p**2 - st.r**2)],
                        [MAV.gamma7 * st.p * st.q - MAV.gamma1 * st.q * st.r]]) + np.array([
                            [MAV.gamma3 * fm.l + MAV.gamma4 * fm.n],
                            [1/MAV.Jy * fm.m],
                            [MAV.gamma4 * fm.l + MAV.gamma8 * fm.n]])
    
    
    # collect the derivative of the states
    x_dot = np.empty( (IND.NUM_STATES,1) )
    x_dot[IND.NORTH] = pned_dot[0]
    x_dot[IND.EAST] = pned_dot[1]
    x_dot[IND.DOWN] = pned_dot[2]
    x_dot[IND.U] = uvw_dot[0]
    x_dot[IND.V] = uvw_dot[1]
    x_dot[IND.W] = uvw_dot[2]
    x_dot[IND.E0] = e_dot[0]
    x_dot[IND.E1] = e_dot[1]
    x_dot[IND.E2] = e_dot[2]
    x_dot[IND.E3] = e_dot[3]
    x_dot[IND.P] = pqrt_dot[0]
    x_dot[IND.Q] = pqrt_dot[1]
    x_dot[IND.R] = pqrt_dot[2]

    return x_dot
