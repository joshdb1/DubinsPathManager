"""
Full Dubin's path manager test
"""
import mav_sim.parameters.planner_parameters as PLAN
import numpy as np
from mav_sim.chap3.mav_dynamics import DynamicState
from mav_sim.chap11.run_sim import run_sim
from mav_sim.message_types.msg_sim_params import MsgSimParams
from mav_sim.message_types.msg_waypoints import MsgWaypoints


def main() -> None:
    """Provide a test scenario for final project (Full Dubin's Path)
    """
    # Initialize the simulation parameters
    sim_params = MsgSimParams(end_time=150., video_name="dubins_path_sim.avi", write_video=False) # Sim ending in 10 seconds
    state = DynamicState()

    R = 136.62760706610337

    # Waypoint definition
    waypoints = MsgWaypoints()
    # waypoints.type = 'straight_line'
    # waypoints.type = 'fillet'
    waypoints.type = 'dubins'
    Va = PLAN.Va0

    waypoints.add(np.array([[0, 0, -100]]).T, Va, np.radians(0), np.inf, 0, 0)
    waypoints.add(np.array([[-3 * R, -R, -100]]).T, Va, np.radians(-90), np.inf, 0, 0)
    waypoints.add(np.array([[0, -800, -100]]).T, Va, np.radians(0), np.inf, 0, 0)
    waypoints.add(np.array([[-3 * R, -800 + R, -100]]).T, Va, np.radians(90), np.inf, 0, 0)
    waypoints.add(np.array([[-4 * R, 1000, -100]]).T, Va, np.radians(45), np.inf, 0, 0)

    # Run the simulation - Note that while not used, the viewer objects
    # need to remain active to keep the windows open
    (waypoint_view, data_view) = run_sim(sim=sim_params, waypoints=waypoints, init_state=state) #pylint: disable=unused-variable

    # Wait until user is finished
    print("Press any key to close")
    input()

if __name__ == "__main__":
    main()
