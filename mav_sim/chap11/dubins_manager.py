"""Provides an implementation of the dubins path manager for waypoint following as described in
   Chapter 11 Algorithms 9 and 10
"""

import numpy as np
from mav_sim.chap11.dubins_parameters import DubinsParameters
from mav_sim.chap11.path_manager_utilities import (
    HalfSpaceParams,
    WaypointIndices,
    get_airspeed,
    inHalfSpace,
)
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.message_types.msg_waypoints import MsgWaypoints

# pylint: disable=too-many-arguments

def dubins_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
                path_prv: MsgPath, hs_prv: HalfSpaceParams, radius: float, manager_state: int, \
                dubins_path_prv: DubinsParameters) \
            -> tuple[MsgPath, HalfSpaceParams, WaypointIndices, int, DubinsParameters]:
    """Update for the Dubins path manager.
       Updates state machine if the MAV enters into the next halfspace.

    Args:
        state: current state of the vehicle
        waypoints: The waypoints to be followed
        ptr_prv: The indices that were being used on the previous iteration (i.e., current waypoint
                 inidices being followed when manager called)
        hs_prv: The previous halfspace being looked for (i.e., the current halfspace when manager called)
        radius: minimum radius circle for the mav
        manager_state: Integer state of the manager
                1: First portion of beginning circle
                2: Second portion of beginning circle, up to H_1
                3: Straight-line segment, up to H_2
                4: First portion of ending circle
                5: Second portion of the ending circle, up to H_3

    Returns:
        path (MsgPath): Path to be followed
        hs (HalfSpaceParams): Half space parameters corresponding to the next change in state
        ptr (WaypointIndices): Indices of the current waypoint being followed
        manager_state (int): The current state of the manager
    """
    # Default the outputs to be the inputs
    path = path_prv
    hs = hs_prv
    ptr = ptr_prv
    dubins_path = dubins_path_prv

    dubins_path.radius = radius

    p = np.array([[state.north, state.east, -state.altitude]]).T

    if waypoints.flag_waypoints_changed is True:
        waypoints.flag_waypoints_changed = False
        ptr = WaypointIndices()
        manager_state = 1

    prev = waypoints.get_waypoint(ptr.previous)
    curr = waypoints.get_waypoint(ptr.current)
    dubins_path = DubinsParameters(p_s=prev.ned, chi_s=prev.course, p_e=curr.ned, chi_e=curr.course, R=radius)

    # Manage the Dubins sections
    if manager_state == 1:
        path, hs = construct_dubins_circle_start(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        hs1 = hs
        hs1.normal = -hs.normal
        hs = hs1
        if inHalfSpace(p, hs1):
            manager_state = 2
    elif manager_state == 2:
        path, hs = construct_dubins_circle_start(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        if inHalfSpace(p, hs):
            if dubins_path.aaa is True:
                manager_state = 4
            else:
                manager_state = 3
    elif manager_state == 3:
        path, hs = construct_dubins_line(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        if inHalfSpace(p, hs):
            manager_state = 6

    # added states for intermediate Dubin's circle
    elif manager_state == 4:
        path, hs = construct_dubins_circle_interm(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        hs1 = hs
        hs1.normal = -hs.normal
        hs = hs1
        if inHalfSpace(p, hs1):
            manager_state = 5
    elif manager_state == 5:
        path, hs = construct_dubins_circle_interm(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        if inHalfSpace(p, hs):
            manager_state = 6

    # back to normal states
    elif manager_state == 6:
        path, hs = construct_dubins_circle_end(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        hs1 = hs
        hs1.normal = -hs.normal
        hs = hs1
        if inHalfSpace(p, hs1):
            manager_state = 7
    elif manager_state == 7:
        path, hs = construct_dubins_circle_end(waypoints=waypoints, ptr=ptr, dubins_path=dubins_path)
        if inHalfSpace(p, hs):
            manager_state = 1
            ptr.increment_pointers(waypoints.num_waypoints)
            # prev = waypoints.get_waypoint(ptr.previous)
            # curr = waypoints.get_waypoint(ptr.current)
            # dubins_path = DubinsParameters(p_s=prev.ned, chi_s=prev.course, p_e=curr.ned, chi_e=curr.course, R=radius)  

    return (path, hs, ptr, manager_state, dubins_path)

def construct_dubins_circle_start(waypoints: MsgWaypoints, ptr: WaypointIndices, dubins_path: DubinsParameters) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """ Create the starting orbit for the dubin's path

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        dubins_Path: The parameters that make-up the Dubin's path between waypoints

    Returns:
        path: The first circle of the Dubin's path
        hs: The halfspace for switching to the next waypoint (H_1)
    """
    # Create the orbit
    path = MsgPath()
    path.type = 'orbit'
    path.plot_updated = False
    path.airspeed = get_airspeed(waypoints, ptr)

    # Fill in remaining parameters
    path.orbit_center = dubins_path.center_s
    path.orbit_direction = "CW" if dubins_path.dir_s == 1 else "CCW"
    path.orbit_radius = dubins_path.radius

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = dubins_path.r1
    hs.normal = dubins_path.n1

    return (path, hs)


def construct_dubins_line(waypoints: MsgWaypoints, ptr: WaypointIndices, dubins_path: DubinsParameters) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """ Create the straight line segment for the dubin's path

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        dubins_Path: The parameters that make-up the Dubin's path between waypoints

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint (H_2)
    """
    # Create the line
    path = MsgPath()
    path.type = 'line'
    path.plot_updated = False
    path.airspeed = get_airspeed(waypoints, ptr)

    # Fill in remaining parameters
    path.line_direction = dubins_path.n1
    path.line_origin = dubins_path.r1

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = dubins_path.r2
    hs.normal = dubins_path.n1

    return (path, hs)


def construct_dubins_circle_end(waypoints: MsgWaypoints, ptr: WaypointIndices, dubins_path: DubinsParameters) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """ Create the ending orbit for the dubin's path

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        dubins_Path: The parameters that make-up the Dubin's path between waypoints

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # Create the orbit
    path = MsgPath()
    path.plot_updated = False
    path.type = 'orbit'
    path.airspeed = get_airspeed(waypoints, ptr)

    # Fill in remaining parameters
    path.orbit_center = dubins_path.center_e
    path.orbit_direction = "CW" if dubins_path.dir_e == 1 else "CCW"
    path.orbit_radius = dubins_path.radius

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = dubins_path.r3
    hs.normal = dubins_path.n3

    return (path, hs)

def construct_dubins_circle_interm(waypoints: MsgWaypoints, ptr: WaypointIndices, dubins_path: DubinsParameters) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """ Create the intermediate orbit for the dubin's path

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        dubins_Path: The parameters that make-up the Dubin's path between waypoints

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # create the orbin
    path = MsgPath()
    path.plot_updated = False
    path.type = 'orbit'
    path.airspeed = get_airspeed(waypoints, ptr)

    # Fill in the remaining parameters
    path.orbit_center = dubins_path.center_i
    path.orbit_direction =  "CW" if dubins_path.dir_i == 1 else "CCW"
    path.orbit_radius = dubins_path.radius

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = dubins_path.r2
    hs.normal = dubins_path.n2

    return (path, hs)
