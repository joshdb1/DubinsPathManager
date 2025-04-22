"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV using Euler coordinates
    - use Euler angles for the attitude state

part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:
        12/17/2018 - RWB
        1/14/2019 - RWB
        12/21 - GND
        12/22 - GND
"""
import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np
from mav_sim.chap3.mav_dynamics import IND, ForceMoments
from mav_sim.tools import types
from mav_sim.tools.rotations import Euler2Quaternion, Euler2Rotation, Quaternion2Euler


# Indexing constants for state using Euler representation
class StateIndicesEuler:
    """Constant class for easy access of state indices
    """
    NORTH: int  = 0  # North position
    EAST: int   = 1  # East position
    DOWN: int   = 2  # Down position
    U: int      = 3  # body-x velocity
    V: int      = 4  # body-y velocity
    W: int      = 5  # body-z velocity
    PHI: int    = 6  # Roll angle (about x-axis)
    THETA: int  = 7  # Pitch angle (about y-axis)
    PSI: int    = 8  # Yaw angle (about z-axis)
    P: int      = 9 # roll rate - body frame - i
    Q: int      = 10 # pitch rate - body frame - j
    R: int      = 11 # yaw rate - body frame - k
    VEL: list[int] = [U, V, W] # Body velocity indices
    ANG_VEL: list[int] = [P, Q, R] # Body rotational velocities
    NUM_STATES: int = 12 # Number of states
IND_EULER = StateIndicesEuler()

# Conversion functions
def euler_state_to_quat_state(state_euler: types.DynamicStateEuler) -> types.DynamicState:
    """Converts an Euler state representation to a quaternion state representation

    Args:
        state_euler: The state vector to be converted to a quaternion representation

    Returns:
        state_quat: The converted state
    """
    # Create the quaternion from the euler coordinates
    e = Euler2Quaternion(phi=state_euler[IND_EULER.PHI], theta=state_euler[IND_EULER.THETA], psi=state_euler[IND_EULER.PSI])

    # Copy over data
    state_quat = np.zeros((IND.NUM_STATES,1))
    state_quat[IND.NORTH] = state_euler.item(IND_EULER.NORTH)
    state_quat[IND.EAST] = state_euler.item(IND_EULER.EAST)
    state_quat[IND.DOWN] = state_euler.item(IND_EULER.DOWN)
    state_quat[IND.U] = state_euler.item(IND_EULER.U)
    state_quat[IND.V] = state_euler.item(IND_EULER.V)
    state_quat[IND.W] = state_euler.item(IND_EULER.W)
    state_quat[IND.E0] = e.item(0)
    state_quat[IND.E1] = e.item(1)
    state_quat[IND.E2] = e.item(2)
    state_quat[IND.E3] = e.item(3)
    state_quat[IND.P] = state_euler.item(IND_EULER.P)
    state_quat[IND.Q] = state_euler.item(IND_EULER.Q)
    state_quat[IND.R] = state_euler.item(IND_EULER.R)

    return state_quat

def quat_state_to_euler_state(state_quat: types.DynamicState) -> types.DynamicStateEuler:
    """Converts a quaternion state representation to an Euler state representation

    Args:
        state_quat: The state vector to be converted

    Returns
        state_euler: The converted state
    """
    # Create the quaternion from the euler coordinates
    phi, theta, psi = Quaternion2Euler(state_quat[IND.QUAT])

    # Copy over data
    state_euler = np.zeros((IND_EULER.NUM_STATES,1))
    state_euler[IND_EULER.NORTH] = state_quat.item(IND.NORTH)
    state_euler[IND_EULER.EAST]  = state_quat.item(IND.EAST)
    state_euler[IND_EULER.DOWN]  = state_quat.item(IND.DOWN)
    state_euler[IND_EULER.U]     = state_quat.item(IND.U)
    state_euler[IND_EULER.V]     = state_quat.item(IND.V)
    state_euler[IND_EULER.W]     = state_quat.item(IND.W)
    state_euler[IND_EULER.PHI]   = phi
    state_euler[IND_EULER.THETA] = theta
    state_euler[IND_EULER.PSI]   = psi
    state_euler[IND_EULER.P]     = state_quat.item(IND.P)
    state_euler[IND_EULER.Q]     = state_quat.item(IND.Q)
    state_euler[IND_EULER.R]     = state_quat.item(IND.R)

    return state_euler


class DynamicStateEuler:
    """Struct for the dynamic state
    """
    def __init__(self, state: types.DynamicStateEuler ) -> None:
        self.north: float     # North position
        self.east: float      # East position
        self.down: float      # Down position
        self.u: float         # body-x velocity
        self.v: float         # body-y velocity
        self.w: float         # body-z velocity
        self.phi: float       # roll angle (about x-axis)
        self.theta: float     # pitch angle (about y-axis)
        self.psi: float       # yaw angle (about z-axis)
        self.p: float         # roll rate - body frame - i
        self.q: float         # pitch rate - body frame - j
        self.r: float         # yaw rate - body frame - k

        self.extract_state(state)

    def extract_state(self, state: types.DynamicStateEuler) ->None:
        """Initializes the state variables

        Args:
            state: State from which to extract the state values

        """
        self.north = state.item(IND_EULER.NORTH)
        self.east =  state.item(IND_EULER.EAST)
        self.down =  state.item(IND_EULER.DOWN)
        self.u =     state.item(IND_EULER.U)
        self.v =     state.item(IND_EULER.V)
        self.w =     state.item(IND_EULER.W)
        self.phi =   state.item(IND_EULER.PHI)
        self.theta = state.item(IND_EULER.THETA)
        self.psi =   state.item(IND_EULER.PSI)
        self.p =     state.item(IND_EULER.P)
        self.q =     state.item(IND_EULER.Q)
        self.r =     state.item(IND_EULER.R)

    def convert_to_numpy(self) -> types.DynamicStateEuler:
        """Converts the state to a numpy object
        """
        output = np.empty( (IND_EULER.NUM_STATES,1) )
        output[IND_EULER.NORTH, 0] = self.north
        output[IND_EULER.EAST, 0] = self.east
        output[IND_EULER.DOWN, 0] = self.down
        output[IND_EULER.U, 0] = self.u
        output[IND_EULER.V, 0] = self.v
        output[IND_EULER.W, 0] = self.w
        output[IND_EULER.PHI, 0] = self.phi
        output[IND_EULER.THETA, 0] = self.theta
        output[IND_EULER.PSI, 0] = self.psi
        output[IND_EULER.P, 0] = self.p
        output[IND_EULER.Q, 0] = self.q
        output[IND_EULER.R, 0] = self.r

        return types.DynamicState( output )

def derivatives_euler(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> types.DynamicStateEuler:
    """Implements the dynamics xdot = f(x, u) where u is the force/moment vector

    Args:
        state: Current state of the vehicle
        forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

    Returns:
        Time derivative of the state ( f(x,u), where u is the force/moment vector )
    """
    st = DynamicStateEuler(state)
    fm = ForceMoments(forces_moments)

    rot_bv = Euler2Rotation(st.phi, st.theta, st.psi)
    pned_dot = rot_bv @ np.array([st.u, st.v, st.w]).transpose()

    uvw_dot = np.array([[st.r * st.v - st.q * st.w], 
                    [st.p * st.w - st.r * st.u],
                    [st.q * st.u - st.p * st.v]]) + (1 / MAV.mass) * np.array([[fm.fx], [fm.fy], [fm.fz]])

    rot_ptp = np.array([[1, np.sin(st.phi) * np.tan(st.theta), np.cos(st.phi) * np.tan(st.theta)],
                       [0, np.cos(st.phi), -np.sin(st.phi)],
                       [0, np.sin(st.phi) * 1 / np.cos(st.theta), np.cos(st.phi) * 1 / np.cos(st.theta)]])
    angles_dot = rot_ptp @ np.array([st.p, st.q, st.r]).transpose()
    
    pqr_dot = np.array([[MAV.gamma1 * st.p * st.q - MAV.gamma2 * st.q * st.r],
                        [MAV.gamma5 * st.p * st.r - MAV.gamma6 * (st.p**2 - st.r**2)],
                        [MAV.gamma7 * st.p * st.q - MAV.gamma1 * st.q * st.r]]) + np.array([
                            [MAV.gamma3 * fm.l + MAV.gamma4 * fm.n],
                            [1/MAV.Jy * fm.m],
                            [MAV.gamma4 * fm.l + MAV.gamma8 * fm.n]])
    

    # collect the derivative of the states
    x_dot = np.empty( (IND_EULER.NUM_STATES,1) )
    x_dot[IND_EULER.NORTH] = pned_dot[0]
    x_dot[IND_EULER.EAST] = pned_dot[1]
    x_dot[IND_EULER.DOWN] = pned_dot[2]
    x_dot[IND_EULER.U] = uvw_dot[0]
    x_dot[IND_EULER.V] = uvw_dot[1]
    x_dot[IND_EULER.W] = uvw_dot[2]
    x_dot[IND_EULER.PHI] = angles_dot[0]
    x_dot[IND_EULER.THETA] = angles_dot[1]
    x_dot[IND_EULER.PSI] = angles_dot[2]
    x_dot[IND_EULER.P] = pqr_dot[0]
    x_dot[IND_EULER.Q] = pqr_dot[1]
    x_dot[IND_EULER.R] = pqr_dot[2]

    return x_dot
