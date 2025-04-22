<h1>UAV Dubin's Path Simulation</h1>

This project was derived from my final project in ECE5330 - Unmanned Aircraft Systems. The project implements a Dubin's path manager for an unmanned aircraft, based on desired waypoints and several state/control variables.

Note that the Dubin's path logic and controls are located in ```mav_sim/chap11``` and other logic, including path following, state updates, etc, are in the other chapter directories.

To see derivations and more information on this project, please see ```DubinsPath_Report.pdf```.

A video of the simulation is also included in ```dubins_path_sim.avi```.


## Creating and activating the mav_venv Python virtual environment

These instructions assume the python virtual environment is located at the same level as the `mav_sim` folder and is named `mav_venv`, so make sure you are in the `mav_sim_python` directory.

To create the virtual environment, type the following command:

```
python -m venv mav_venv
```
You may need to replace `python` with `python3` or `python3.10`.

Now activate the virtual environment with the following command.

```
Windows:
mav_venv\Scripts\activate

Ubuntu:
source mav_venv/bin/activate
```

## Installing the `mav_sim` package

To install the `mav_sim` package, make sure you are in the `mav_sim_python` directory use the following command:
```
pip install -e .
```
Wait a few minutes for the installation to complete.

## Running the Full Dubin's Path Manager

The Dubin's Path Manager test is located in `dubins_path_sim.py`. To run it, simply run the command:

```
python .\dubins_path_sim.py
```

The simulation window will pop up and the aircraft will begin flying the path specified.
