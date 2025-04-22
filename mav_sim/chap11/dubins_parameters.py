"""
 dubins_parameters
   - Dubins parameters that define path between two configurations

 mavsim_matlab
     - Beard & McLain, PUP, 2012
     - Update history:
         3/26/2019 - RWB
         4/2/2020 - RWB
         12/21 - GND
"""

# from typing import cast

import numpy as np
from mav_sim.tools import types
from mav_sim.tools.rotations import rotz  # Function for rotation about z-axis
from mav_sim.tools.types import NP_MAT  # Type for a numpy matrix


class DubinsParamsStruct:
    """Class for passing out calculated parameters
    """

    __slots__ = [
        "L",
        "c_s",
        "lam_s",
        "c_e",
        "lam_e",
        "c_i",
        "lam_i",
        "z1",
        "q1",
        "z2",
        "q2",
        "z3",
        "q3",
        "aaa",
    ]
    def __init__(self) -> None:
        self.L: float           # path length
        self.c_s: types.NP_MAT  # starting circle center
        self.lam_s: int         # direction of the start circle (+1 for CW/right, -1 for CCW/left)
        self.c_e: types.NP_MAT  # ending circle center
        self.lam_e: int         # direction of the end circle (+1 for CW/right, -1 for CCW/left)
        self.c_i: types.NP_MAT  # intermediate circle center
        self.lam_i: int         # direction of the interm circle (+1 for CW/right, -1 for CCW/left)
        self.z1: types.NP_MAT   # Point on halfspace 1 boundary
        self.q1: types.NP_MAT   # Normal vector for halfspace 1
        self.z2: types.NP_MAT   # Point on halfspace 2 boundary (note that normal vector is same as halfpace 1)
        self.q2: types.NP_MAT
        self.z3: types.NP_MAT   # Point on halfspace 3 boundary
        self.q3: types.NP_MAT   # Normal vector for halfspace 3
        self.aaa: bool          # flag to use an arc-arc-arc type

    def print(self) -> None:
        """Print the commands to the console."""
        print(
            "\n=L",self.L,
            "\n=c_s",self.c_s,
            "\n=lam_s",self.lam_s,
            "\n=c_e",self.c_e,
            "\n=lam_e",self.lam_e,
            "\n=c_i",self.c_i,
            "\n=lam_i",self.lam_i,
            "\n=z1",self.z1,
            "\n=q1",self.q1,
            "\n=z2",self.z2,
            "\n=q2",self.q2,
            "\n=z3",self.z3,
            "\n=q3",self.q3,
            "\n=aaa",self.aaa,
        )

class DubinsPoints:
    """Struct for storing points and radius used for calculating Dubins path
    """
    __slots__ = [
        "p_s",
        "chi_s",
        "p_e",
        "chi_e",
        "radius",
    ]
    def __init__(self, p_s: NP_MAT, chi_s: float, p_e: NP_MAT, chi_e: float, radius: float) -> None:
        self.p_s = p_s          # starting position
        self.chi_s = chi_s      # starting course angle
        self.p_e = p_e          # ending position
        self.chi_e = chi_e      # ending course angle
        self.radius = radius    # radius of Dubin's paths arcs

    def extract(self) -> tuple[NP_MAT, float, NP_MAT, float, float]:
        """Extracts all of the elements into a tuple
        """
        return (self.p_s, self.chi_s, self.p_e, self.chi_e, self.radius)

class DubinsParameters:
    """Class for storing the parameters for a Dubin's path
    """
    def __init__(self, p_s: NP_MAT =9999*np.ones((3,1)), chi_s: float =9999,
                 p_e: NP_MAT =9999*np.ones((3,1)), chi_e: float =9999, R: float =9999) -> None:
        """ Initialize the Dubin's path

        Args:
            p_s: starting position
            chi_s: starting course angle
            p_e: ending position
            chi_e: ending course angle
            R: radius of Dubin's path arcs
        """
        # Store input parameters
        self.p_s = p_s      # starting position
        self.chi_s = chi_s  # starting course angle
        self.p_e = p_e      # ending position
        self.chi_e = chi_e  # ending course angle
        self.radius = R     # radius of Dubin's paths arcs

        # Initialize calculated parameters
        self.length: float          # Dubin's path length
        self.center_s: types.NP_MAT # starting circle center
        self.dir_s: float           # direction of the start circle (1 => "CW", and "CCW" otherwise)
        self.center_e: types.NP_MAT # ending circle center
        self.dir_e: float           # direction of the end circle (1 => "CW", and "CCW" otherwise)
        self.center_i: types.NP_MAT # intermediate circle center
        self.dir_i: float           # direction of the interm circle (1 => "CW", and "CCW" otherwise)
        self.r1: types.NP_MAT       # Point on halfspace 1
        self.n1: types.NP_MAT       # Normal vector for halfspace 1
        self.r2: types.NP_MAT       # Point on halfspace 2 (note that normal vector is same as halfpace 1)
        self.n2: types.NP_MAT
        self.r3: types.NP_MAT       # Point on halfspace 3
        self.n3: types.NP_MAT       # Normal vector for halfspace 3
        self.aaa: bool              # Flag to use arc-arc-arc

        if R == 9999: # Infinite radius case - straight line
            dubin = DubinsParamsStruct()
            dubin.L = R
            dubin.c_s = p_s
            dubin.lam_s = 1
            dubin.c_e = p_s
            dubin.lam_e = 1
            dubin.c_i = p_s
            dubin.lam_i = 1
            dubin.z1 = p_s
            dubin.q1 = p_s
            dubin.z2 = p_s
            dubin.q2 = p_s
            dubin.z3 = p_s
            dubin.q3 = p_s
            dubin.aaa = False
        else:
            points = DubinsPoints(p_s=p_s, chi_s=chi_s, p_e=p_e, chi_e=chi_e, radius=R)
            dubin = compute_parameters(points)
        self.set(dubin)

    def set(self, vals: DubinsParamsStruct) -> None:
        """Sets the class variables based upon the Dubins parameter struct

        Args:
            vals: Values to be stored in the class
        """
        self.length = vals.L         # Dubin's path length
        self.center_s = vals.c_s     # starting circle center
        self.dir_s = vals.lam_s      # direction of the start circle (1 => "CW", and "CCW" otherwise)
        self.center_e = vals.c_e     # ending circle center
        self.dir_e = vals.lam_e      # direction of the end circle (1 => "CW", and "CCW" otherwise)
        self.center_i = vals.c_i     # interm circle center
        self.dir_i = vals.lam_i      # direction of the interm circle (1 => "CW", and "CCW" otherwise)
        self.r1 = vals.z1            # Point on halfspace 1
        self.n1 = vals.q1            # Normal vector for halfspace 1
        self.r2 = vals.z2            # Point on halfspace 2 (note that normal vector is same as halfpace 1)
        self.n2 = vals.q2            # Normal vector for halfspace 2
        self.r3 = vals.z3            # Point on halfspace 3
        self.n3 = vals.q3            # Normal vector for halfspace 3
        self.aaa = vals.aaa

def compute_parameters(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculate the dubins paths parameters. Returns the parameters defining the shortest
       path between two oriented waypoints

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the shortest Dubins path
    """

    # Check to ensure sufficient distance between points
    # (p_s, _, p_e, _, R) = points.extract()
    # ell = np.linalg.norm(p_s[0:2] - p_e[0:2])
    # if ell < 2 * R:
    #     raise ValueError('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()

    dubin_rsr = calculate_rsr(points)
    dubin_rsl = calculate_rsl(points)
    dubin_lsr = calculate_lsr(points)
    dubin_lsl = calculate_lsl(points)
    dubin_rlr = calculate_rlr(points)
    dubin_lrl = calculate_lrl(points)

    dubins = [dubin_rsr, dubin_rsl, dubin_lsr, dubin_lsl, dubin_rlr, dubin_lrl]
    dists = np.array([dubin_rsr.L, dubin_rsl.L, dubin_lsr.L, dubin_lsl.L, dubin_rlr.L, dubin_lrl.L])

    minIdx = np.argmin(dists)

    dubin = dubins[minIdx]

    return dubin

def calculate_rsr(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the right-straight-right case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    rot = rotz(-np.pi / 2)

    # Calculate distance and switching surfaces
    dubin.c_s = p_s + R * np.array([[np.cos(chi_s + np.pi / 2), np.sin(chi_s + np.pi / 2), 0]]).T
    dubin.lam_s = 1
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e + np.pi / 2), np.sin(chi_e + np.pi / 2), 0]]).T
    dubin.lam_e = 1

    dubin.c_i = np.array([[0, 0, 0]]).T
    dubin.lam_i = 0

    dubin.aaa = False

    l = np.linalg.norm(dubin.c_e - dubin.c_s)

    if l >= 2 * R:

        dubin.q1 = (dubin.c_e - dubin.c_s) / np.linalg.norm(dubin.c_e - dubin.c_s)
        dubin.z1 = dubin.c_s + R * rot @ dubin.q1
        dubin.z2 = dubin.c_e + R * rot @ dubin.q1
        dubin.q2 = dubin.q1
        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ np.array([[1, 0, 0]]).T

        var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))
    
        dubin.L = np.linalg.norm(dubin.c_s - dubin.c_e).item(0) + \
            R * (mod(2 * np.pi + (var - np.pi / 2) - (chi_s - np.pi / 2))) + \
            R * mod(2 * np.pi + (chi_e - np.pi / 2) - (var - np.pi / 2))
    else:
        dubin.L = np.inf

    return dubin

def calculate_rsl(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the right-straight-left case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()
        
    # Calculate distance and switching surfaces
    # dubin.L = 99999.
    dubin.c_s = p_s + R * np.array([[np.cos(chi_s + np.pi / 2), np.sin(chi_s + np.pi / 2), 0]]).T
    dubin.lam_s = 1
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e - np.pi / 2), np.sin(chi_e - np.pi / 2), 0]]).T
    dubin.lam_e = -1

    dubin.c_i = np.array([[0, 0, 0]]).T
    dubin.lam_i = 0
    
    dubin.aaa = False

    l = np.linalg.norm(dubin.c_s - dubin.c_e).item(0)

    if l >= 2 * R:
        var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))
        var2 = var - np.pi / 2 + np.arcsin(2 * R / l)
        e1 = np.array([[1, 0, 0]]).T

        dubin.q1 = rotz(var2 + np.pi / 2) @ e1
        dubin.z1 = dubin.c_s + R * rotz(var2) @ e1
        dubin.z2 = dubin.c_e + R * rotz(var2 + np.pi) @ e1
        dubin.q2 = dubin.q1
        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ e1

        dubin.L = np.sqrt(l**2 - 4 * R**2) + \
            R * mod(2 * np.pi + var2 - (chi_s - np.pi / 2)) + \
            R * mod(2 * np.pi + (var2 + np.pi) - (chi_e + np.pi / 2))
    else:
        dubin.L = np.inf

    return dubin

def calculate_lsr(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the left-straight-right case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    # Calculate distance and switching surfaces
    # dubin.L = 99999.
    dubin.c_s = p_s + R * np.array([[np.cos(chi_s - np.pi / 2), np.sin(chi_s - np.pi / 2), 0]]).T
    dubin.lam_s = -1
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e + np.pi / 2), np.sin(chi_e + np.pi / 2), 0]]).T
    dubin.lam_e = 1

    dubin.c_i = np.array([[0, 0, 0]]).T
    dubin.lam_i = 0
    
    dubin.aaa = False

    l = np.linalg.norm(dubin.c_s - dubin.c_e).item(0)

    if l >= 2 * R:
        var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))
        var2 = np.arccos(2 * R / l)
        e1 = np.array([[1, 0, 0]]).T

        dubin.q1 = rotz(var + var2 - np.pi / 2) @ e1
        dubin.z1 = dubin.c_s + R * rotz(var + var2) @ e1
        dubin.z2 = dubin.c_e + R * rotz(var + var2 - np.pi) @ e1
        dubin.q2 = dubin.q1
        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ e1

    
        dubin.L = np.sqrt(l**2 - 4 * R**2) + \
            R * mod(2 * np.pi + (chi_s + np.pi / 2) - (var + var2)) + \
            R * mod(2 * np.pi + (chi_e - np.pi / 2) - (var + var2 - np.pi))
    else:
        dubin.L = np.inf

    return dubin

def calculate_lsl(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the left-straight-left case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

        # Calculate distance and switching surfaces
    dubin.c_s = p_s + R * np.array([[np.cos(chi_s - np.pi / 2), np.sin(chi_s - np.pi / 2), 0]]).T
    dubin.lam_s = -1
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e - np.pi / 2), np.sin(chi_e - np.pi / 2), 0]]).T
    dubin.lam_e = -1

    dubin.c_i = np.array([[0, 0, 0]]).T
    dubin.lam_i = 0
    
    dubin.aaa = False

    l = np.linalg.norm(dubin.c_s - dubin.c_e).item(0)
    if l >= 2 * R:

        dubin.q1 = (dubin.c_e - dubin.c_s) / np.linalg.norm(dubin.c_e - dubin.c_s)
        dubin.z1 = dubin.c_s + R * rotz(np.pi / 2) @ dubin.q1
        dubin.z2 = dubin.c_e + R * rotz(np.pi / 2) @ dubin.q1
        dubin.q2 = dubin.q1
        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ np.array([[1, 0, 0]]).T

        var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))

        dubin.L = np.linalg.norm(dubin.c_s - dubin.c_e).item(0) + \
            R * mod(2 * np.pi + (chi_s + np.pi / 2) - (var + np.pi / 2)) + \
            R * mod(2 * np.pi + (var + np.pi / 2) - (chi_e + np.pi / 2))
    else:
        dubin.L = np.inf

    return dubin

def calculate_rlr(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the right-left-right case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    dubin.c_s = p_s + R * np.array([[np.cos(chi_s + np.pi / 2), np.sin(chi_s + np.pi / 2), 0]]).T
    dubin.lam_s = 1
    
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e + np.pi / 2), np.sin(chi_e + np.pi / 2), 0]]).T
    dubin.lam_e = 1

    dubin.aaa = True

    v = dubin.c_e - dubin.c_s

    var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))

    if np.linalg.norm(dubin.c_s - dubin.c_e).item(0) < 4 * R:
        # compute the distance from start to end circle centers
        D = np.linalg.norm(dubin.c_e - dubin.c_s)
        
        # compute the angle from intermediate center and start/end centers
        theta = (var - np.arccos(D / (4 * R))).item(0)
        

        # compute center of intermediate circle
        dubin.c_i = np.array([[dubin.c_s.item(0) + 2 * R * np.cos(theta),
                            dubin.c_s.item(1) + 2 * R * np.sin(theta), 
                            dubin.c_s.item(2)]]).T
        dubin.lam_i = -1

        # calculate half spaces
        dubin.z1 = dubin.c_i + (dubin.c_s - dubin.c_i) / np.linalg.norm(dubin.c_s - dubin.c_i) * R

        v = dubin.c_s - dubin.c_i
        dubin.q1 = np.array([[v.item(1), -v.item(0), 0]]).T
        dubin.q1 = dubin.q1 / np.linalg.norm(dubin.q1)

        dubin.z2 = dubin.c_i + (dubin.c_e - dubin.c_i) / np.linalg.norm(dubin.c_e - dubin.c_i) * R

        v = dubin.c_e - dubin.c_i
        dubin.q2 = np.array([[v.item(1), -v.item(0), 0]]).T
        dubin.q2 = dubin.q2 / np.linalg.norm(dubin.q2)

        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ np.array([[1, 0, 0]]).T

        # d = np.linalg.norm(dubin.z1 - p_s)
        # var1 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_s - p_s) / np.linalg.norm(dubin.c_s - p_s)
        b = (dubin.c_s - dubin.z1) / np.linalg.norm(dubin.c_s - dubin.z1)
        var1 = np.arccos(a.T @ b).item(0)
        
        # d = np.linalg.norm(dubin.z2 - dubin.z1)
        # var2 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_i - dubin.z1) / np.linalg.norm(dubin.c_i - dubin.z1)
        b = (dubin.c_i - dubin.z2) / np.linalg.norm(dubin.c_i - dubin.z2)
        var2 = np.arccos(a.T @ b).item(0)

        # d = np.linalg.norm(p_e - dubin.z2)
        # var3 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_e - dubin.z2) / np.linalg.norm(dubin.c_e - dubin.z2)
        b = (dubin.c_e - p_e) / np.linalg.norm(dubin.c_e - p_e)
        var3 = np.arccos(a.T @ b).item(0)

        dubin.L = R * mod(var1) + R * mod(var2) + R * mod(var3)
    else:
        dubin.L = np.inf

    return dubin

def calculate_lrl(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the left-right-left case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    dubin.c_s = p_s + R * np.array([[np.cos(chi_s - np.pi / 2), np.sin(chi_s - np.pi / 2), 0]]).T
    dubin.lam_s = -1
    dubin.c_e = p_e + R * np.array([[np.cos(chi_e - np.pi / 2), np.sin(chi_e - np.pi / 2), 0]]).T
    dubin.lam_e = -1

    dubin.aaa = True

    v = dubin.c_e - dubin.c_s

    var = np.arctan2(dubin.c_e.item(1) - dubin.c_s.item(1), dubin.c_e.item(0) - dubin.c_s.item(0))

    if np.linalg.norm(dubin.c_s - dubin.c_e).item(0) < 4 * R:
        # compute the distance from start to end circle centers
        D = np.linalg.norm(dubin.c_e - dubin.c_s)
        
        # compute the angle from intermediate center and start/end centers
        theta = (var + np.arccos(D / (4 * R))).item(0)

        # compute center of intermediate circle
        dubin.c_i = np.array([[dubin.c_s.item(0) + 2 * R * np.cos(theta),
                            dubin.c_s.item(1) + 2 * R * np.sin(theta), 
                            dubin.c_s.item(2)]]).T
        dubin.lam_i = 1

        # calculate half spaces
        dubin.z1 = dubin.c_i + (dubin.c_s - dubin.c_i) / np.linalg.norm(dubin.c_s - dubin.c_i) * R

        v = dubin.c_s - dubin.c_i
        dubin.q1 = np.array([[-v.item(1), v.item(0), 0]]).T
        dubin.q1 = dubin.q1 / np.linalg.norm(dubin.q1)

        dubin.z2 = dubin.c_i + (dubin.c_e - dubin.c_i) / np.linalg.norm(dubin.c_e - dubin.c_i) * R
        
        v = dubin.c_e - dubin.c_i
        dubin.q2 = np.array([[-v.item(1), v.item(0), 0]]).T
        dubin.q2 = dubin.q2 / np.linalg.norm(dubin.q2)

        dubin.z3 = p_e
        dubin.q3 = rotz(chi_e) @ np.array([[1, 0, 0]]).T

        # d = np.linalg.norm(dubin.z1 - p_s)
        # var1 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_s - p_s) / np.linalg.norm(dubin.c_s - p_s)
        b = (dubin.c_s - dubin.z1) / np.linalg.norm(dubin.c_s - dubin.z1)
        var1 = np.arccos(a.T @ b).item(0)
        
        # d = np.linalg.norm(dubin.z2 - dubin.z1)
        # var2 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_i - dubin.z1) / np.linalg.norm(dubin.c_i - dubin.z1)
        b = (dubin.c_i - dubin.z2) / np.linalg.norm(dubin.c_i - dubin.z2)
        var2 = np.arccos(a.T @ b).item(0)

        # d = np.linalg.norm(p_e - dubin.z2)
        # var3 = np.arccos((2 * R**2 - d**2) / (2 * R**2))
        a = (dubin.c_e - dubin.z2) / np.linalg.norm(dubin.c_e - dubin.z2)
        b = (dubin.c_e - p_e) / np.linalg.norm(dubin.c_e - p_e)
        var3 = np.arccos(a.T @ b).item(0)

        dubin.L = R * mod(var1) + R * mod(var2) + R * mod(var3)
    else:
        dubin.L = np.inf

    return dubin

def mod(x: float) -> float:
    """Computes the modulus of x with respect to 2 pi

    Args:
        x: Angle

    Returns:
        x: Angle modified to be between 0 and 2pi
    """
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x
