<h1>UAV Dubin's Path Simulation</h1>

This project was derived from my final project in ECE5330 - Unmanned Aircraft Systems. The project implements a Dubin's path manager for an unmanned aircraft, based on desired waypoints and several state/control variables.

Note that the Dubin's path logic and controls are located in ```mav_sim/chap11``` and other logic, including path following, state updates, etc, are in the other chapter directories.

To see derivations and more information on this project, please see ```DubinsPath_Report.pdf```.

A video of the simulation is also included in ```dubins_path_sim.avi```.


<h2>Installation</h2>
To run this simulation, you must have Python3.10+ installed and install the dependencies, including MavSim. Creating a virtual environment is recommended.

If make is installed, you can run ```make install``` 
Otherwise, run ```pip install -r requirements.txt```

<h2>Running</h2>
To run the Dubin's path simulation, run ```python dubins_path_sim.py```
This should bring up two MavSim windows showing the aircraft's position on the path and control states.

<h2>MavSim Documentation</h2>

For documentation on the python simulation, see [mavsim_python/README.md](mavsim_python/README.md)
