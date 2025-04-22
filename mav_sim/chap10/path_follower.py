"""path_follower.py implements a class for following a path with a mav
"""
# from math import cos, sin

import numpy as np
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools.wrap import wrap


class PathFollower:
    """Class for path following
    """
    def __init__(self) -> None:
        """Initialize path following class
        """
        self.chi_inf = np.radians(50)  # approach angle for large distance from straight-line path
        self.k_path = 0.01 #0.05  # path gain for straight-line path following
        self.k_orbit = 1.# 10.0  # path gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path: MsgPath, state: MsgState) -> MsgAutopilot:
        """Update the control for following the path

        Args:
            path: path to be followed
            state: current state of the mav

        Returns:
            autopilot_commands: Commands to autopilot for following the path
        """
        if path.type == 'line':
            self.autopilot_commands = follow_straight_line(path=path, state=state, k_path=self.k_path, chi_inf=self.chi_inf)
        elif path.type == 'orbit':
            self.autopilot_commands = follow_orbit(path=path, state=state, k_orbit=self.k_orbit, gravity=self.gravity)
        return self.autopilot_commands

def follow_straight_line(path: MsgPath, state: MsgState, k_path: float, chi_inf: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a straight line

    Args:
        path: straight-line path to be followed
        state: current state of the mav
        k_path: convergence gain for converging to the path
        chi_inf: Angle to take towards path when at an infinite distance from the path

    Returns:
        autopilot_commands: the commands required for executing the desired line
    """
    # Initialize the output
    autopilot_commands = MsgAutopilot()

    # course command
    # chi_q = np.arccos(path.line_direction.item(0) / np.linalg.norm(path.line_direction[0:2]))
    chi_q = wrap(np.arctan2(path.line_direction.item(1), path.line_direction.item(0)), state.chi)
    R = np.array([[np.cos(chi_q), np.sin(chi_q), 0], [-np.sin(chi_q), np.cos(chi_q), 0], [0, 0, 1]])

    # epi = R @ (state.p - path.line_origin)
    ep = np.array([[state.north], [state.east], [-state.altitude]]) - path.line_origin
    epi = R @ ep
    epy = epi.item(1)

    # altitude command
    # ep = (state.p - path.line_origin)
    # ep = (np.array([[state.north], [state.east], [-state.altitude]]) - path.line_origin)
    qcrossk = np.array([[path.line_direction.item(1)], [-path.line_direction.item(0)], [0.0]])
    n = qcrossk / np.linalg.norm(qcrossk)
    s = ep - (ep.transpose() @ n) * n
    hd = -path.line_origin.item(2) - np.sqrt(s.item(0)**2 + s.item(1)**2) * (path.line_direction.item(2) \
                    / (np.sqrt(path.line_direction.item(0)**2 + path.line_direction.item(1)**2)))

    # Create autopilot commands here
    autopilot_commands.airspeed_command = path.airspeed
    autopilot_commands.altitude_command = hd
    autopilot_commands.course_command = chi_q - chi_inf * 2 / np.pi * np.arctan2(k_path * epy, 1)
    autopilot_commands.phi_feedforward = 0.0

    return autopilot_commands


def follow_orbit(path: MsgPath, state: MsgState, k_orbit: float, gravity: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a circular path

    Args:
        path: circular orbit to be followed
        state: current state of the mav
        k_orbit: Convergence gain for reducing error to orbit
        gravity: Gravity constant

    Returns:
        autopilot_commands: the commands required for executing the desired orbit
    """

    # Initialize the output
    autopilot_commands = MsgAutopilot()

    lam = 1
    if path.orbit_direction == 'CCW':
        lam = -1

    # course command
    phi = wrap(np.arctan2(state.east - path.orbit_center.item(1), state.north - path.orbit_center.item(0)), state.chi)
    d = np.sqrt((state.north - path.orbit_center.item(0))**2 + (state.east - path.orbit_center.item(1))**2)

    # Create autopilot commands
    autopilot_commands.airspeed_command = path.airspeed
    autopilot_commands.course_command = phi + lam * (np.pi / 2 + np.arctan2(k_orbit * (d - path.orbit_radius), path.orbit_radius))
    autopilot_commands.altitude_command = -path.orbit_center.item(2)

    if (d - path.orbit_radius) / path.orbit_radius < 10:
        autopilot_commands.phi_feedforward = lam * np.arctan2(state.Va**2, gravity * path.orbit_radius)
    else:
        autopilot_commands.phi_feedforward = 0

    return autopilot_commands
