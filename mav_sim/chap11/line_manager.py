"""Provides an implementation of the straight line path manager for waypoint following as described in
   Chapter 11 Algorithm 7
"""

# from typing import cast

import numpy as np
from mav_sim.chap11.path_manager_utilities import (
    HalfSpaceParams,
    WaypointIndices,
    extract_waypoints,
    get_airspeed,
    inHalfSpace,
)
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.message_types.msg_waypoints import MsgWaypoints
# from mav_sim.tools.types import NP_MAT


def line_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
                 path_prv: MsgPath, hs_prv: HalfSpaceParams) \
                -> tuple[MsgPath, HalfSpaceParams, WaypointIndices]:
    """Update for the line manager. Only updates the path and next halfspace under two conditions:
        1) The waypoints are new
        2) In a new halfspace

    Args:
        state: current state of the vehicle
        waypoints: The waypoints to be followed
        ptr_prv: The indices of the waypoints being used for the previous path
        path_prv: The previously commanded path
        hs_prv: The currently active halfspace for switching

    Returns:
        path: The updated path to follow
        hs: The updated halfspace for the next switch
        ptr: The updated index pointer
    """
    # Default the outputs to be the inputs
    path = path_prv
    hs = hs_prv
    ptr = ptr_prv

    # Create manager here
    if waypoints.flag_waypoints_changed is True:
        waypoints.flag_waypoints_changed = False
        ptr = WaypointIndices()

    path, hs = construct_line(waypoints=waypoints, ptr=ptr)

    pos = np.array([[state.north, state.east, -state.altitude]]).T
    if inHalfSpace(pos=pos, hs=hs):
        ptr.increment_pointers(waypoints.num_waypoints)

    # Output the updated path, halfspace, and index pointer
    return (path, hs, ptr)

def construct_line(waypoints: MsgWaypoints, ptr: WaypointIndices) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Creates a line and switching halfspace. The halfspace assumes that the aggregate
       path will consist of a series of straight lines.

    The line is created from the previous and current waypoints with halfspace defined for
    switching once the current waypoint is reached.

    Args:
        waypoints: The waypoints from which to construct the path
        ptr: The indices of the waypoints being used for the path

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """

    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)

    qim1 = (current - previous) / np.linalg.norm(current - previous)
    qi = (next_wp - current) / np.linalg.norm(next_wp - current)

    # Construct the path
    path = MsgPath()
    path.plot_updated = False
    path.line_direction = qim1
    path.line_origin = previous
    path.airspeed = get_airspeed(waypoints, ptr)

    # Construct the halfspace
    hs = HalfSpaceParams()
    if np.isclose(np.linalg.norm(qim1 + qi), 0):
        hs.normal = qim1
        hs.point = current    
    else:
        hs.normal = (qim1 + qi) / np.linalg.norm(qim1 + qi)
        hs.point = current

    return (path, hs)
