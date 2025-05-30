"""
AERO4701 Space Engineering 3
Lunar Atmospheric Investigations with CubeSats (LUNATICS)

A class and functions to simulate a sun sensor system on a CubeSat.
"""

# MODULES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CLASS
class SunSensor:
    def __init__(self, number, face, position_vector, phi, theta, psi):
        """SunSensor class to represent a sun sensor on a CubeSat.
        Args:
            number (int): The identifying number of the sensor
            face (str): The face of the CubeSat where the sensor is mounted.
            position_vector (list): The position vector of the sensor in the body frame.
            phi (float): The roll angle in degrees.
            theta (float): The pitch angle in degrees.
            psi (float): The yaw angle in degrees.
        """

        # Identifying the sensor
        self.number = number
        self.face = face

        # Position vector of the sensor in the body frame
        self.pos_vec = np.array(position_vector)

        # Rotation matrix from the body frame to sensor frame
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        psi = np.deg2rad(psi)
        
        self.rotmat_SB = np.array([
            [np.cos(theta)*np.cos(psi), -np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)],
            [np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)],
            [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]
        ])
        
        self.rotmat_BS = np.linalg.inv(self.rotmat_SB)

        self.z_in_body = self.rotmat_BS @ np.array([0, 0, 1])


    def transform_to_sensor(self, V):
        """Transform a vector from the body frame to the sensor frame.
        Args:
            V (np array): The vector to be transformed, in the body frame, V = [X,Y,Z]
        Returns:
            v (np array): The vector transformed into the local sensor frame, v = [x,y,z]
        """

        V = np.array(V)
        v = self.rotmat_SB @ V 
        print(f"Vector in Body frame: {V}")
        print(f"Vector in Sensor {self.number} ({self.face}) frame: {v}")
        return v
    
    def transform_to_body(self, v):
        """Transform a vector from the sensor frame to the body frame.
        Args:
            v (np array): The vector to be transformed, in the sensor frame, v = [x,y,z]
        Returns:
            V (np array): The vector transformed into the body frame, V = [X,Y,Z]
        """
        
        v = np.array(v)
        V = self.rotmat_BS @ v
        print(f"Vector in Sensor {self.number} ({self.face}) frame: {v}")
        print(f"Vector in Body frame: {V}")
        return V       

    def get_normal(self):
        """Get the normal vector (z-axis, cone axis) of the sensor in the body frame."""
        return self.z_in_body

# Find Sun Vector
def find_sun(sensors, readings, X0, 
             Srel_tol = 0.1, max_iter=1000, tol=1e-6, learning_rate=0.1):
    """
    Args:
        sensors (list): List of SunSensor objects.
        reading (np array): The S_rel readings for each sensor.
        X0 (np array): The initial guess for direction of Sun.
        Srel_tol (float): Tolerance for "0" sensor readings.
        max_iter (int): The maximum number of iterations for NLLS.
        tol (float): Tolerance for minimisation function in NLLS.
        learning_rate (float): The learning rate for NLLS.

    Returns:
        Xhat (np array): Eestimated direction vec of the Sun in body frame.
        selected_sensors (list): Storage of sensors utilised for estimate.
    """

    # Select the top three sensors based on readings
    sorted_idx = np.argsort(readings)[-3:]
    sensors = [sensors[i] for i in sorted_idx]
    readings = readings[sorted_idx]

    # Check three sensors actually have values
    if (readings[0] < Srel_tol) or (readings[1] < Srel_tol) or (readings[2] < Srel_tol):
        print("Not enough valid readings from sensors.")
        return None, [None, None, None]
    else:
        selected_sensors = [sorted_idx[i]+1 for i in sorted_idx]
        
    # Design matrix (H) [3x3]
    H = np.array([sensor.get_normal() for sensor in sensors])

    # Weight matrix (W) [3x3]
    W = np.diag(readings**2)
    W /= np.trace(W)                    # Normalise

    # Observation vector (Y) [3x1]
    Y = readings

    # Initial Sun direction vector estimate
    Xhat = X0 / np.linalg.norm(X0)      # Normalise

    # Non-linear Least Squares Estimation
    iter_count = 0
    for iter in range(max_iter):
        # Update iteration counter
        iter_count += 1

        # Compute residuals
        r = Y - np.dot(H, Xhat)

        # Compute gradient
        gradient = -2 * np.dot(H.T, W @ r)

        # Direction update
        dX = learning_rate * gradient

        # Check if update is significant enough to continue
        if np.sum(np.abs(dX)) > tol:
            # Update direction vector
            Xhat -= dX
            Xhat /= np.linalg.norm(Xhat)    # Normalise
        else:
            break
            
        print(f"Estimated Sun direction vector: {Xhat}")

        return Xhat, selected_sensors
    
# Model the sun sensors on the CubeSat
def model_sun_sensors(sensors, sat_dim):
    """
    Visualises the CubeSat and sun sensors in 3D.

    Args:
        sensors (list): List of SunSensor objects
        sat_dim (list): Dimensions of CubeSat [xdim, ydim, zdim] in metres
    """

    def set_axes_equal(ax):
        """Set 3D plot axes to equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    xdim, ydim, zdim = sat_dim

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("CubeSat Sun Sensor Layout")

    # Draw the CubeSat body as a wireframe cube
    def draw_cube(xdim, ydim, zdim):
        r = [0, xdim]
        for s, e in combinations(np.array(list(product(r, [0, ydim], [0, zdim]))), 2):
            if np.sum(np.abs(s - e)) == ydim or np.sum(np.abs(s - e)) == zdim:
                ax.plot3D(*zip(s, e), color="black", lw=0.5)
        for s, e in combinations(np.array(list(product([0, xdim], r, [0, zdim]))), 2):
            if np.sum(np.abs(s - e)) == xdim or np.sum(np.abs(s - e)) == zdim:
                ax.plot3D(*zip(s, e), color="black", lw=0.5)
        for s, e in combinations(np.array(list(product([0, xdim], [0, ydim], r))), 2):
            if np.sum(np.abs(s - e)) == xdim or np.sum(np.abs(s - e)) == ydim:
                ax.plot3D(*zip(s, e), color="black", lw=0.5)

    from itertools import product, combinations
    draw_cube(xdim, ydim, zdim)

    axis_length = 0.05 * min(sat_dim)  # scale local axes

    for sensor in sensors:
        pos = sensor.pos_vec

        # Plot sensor position
        ax.scatter(*pos, label=f"Sensor {sensor.number} ({sensor.face})")

        # Plot local sensor frame axes (x, y, z)
        origin = pos
        axes_sensor = np.eye(3) * axis_length  # unit vectors scaled
        axes_body = sensor.rotmat_BS @ axes_sensor  # transform into body frame

        colours = ['r', 'g', 'b']
        for i in range(3):
            ax.quiver(*origin, *axes_body[:, i], color=colours[i], arrow_length_ratio=0.2)

    ax.set_xlim([0, xdim])
    ax.set_ylim([0, ydim])
    ax.set_zlim([0, zdim])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    set_axes_equal(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Expected readings
def expected_readings(sensors, sun_vec, exact=True, noise=0.01, max_FOV=80):
    """Compute the expected readings of the sensors based on the sun direction vector.
    Args:
        sensors (list): List of SunSensor objects.
        sun_vec (np array): The sun direction vector in the body frame.
        exact (bool): If True, the exact readings are computed. If False, the readings are affected by noise.
        noise (float): The magnitude of noise to be added to the readings.
        max_FOV (float): The maximum field of view of the sensors (degrees).
    Returns:
        readings (np array): The expected readings of the sensors.
    """

    # Convert max_FOV to radians
    max_FOV = np.deg2rad(max_FOV)

    # Normalise sun direction vector
    sun_vec = sun_vec / np.linalg.norm(sun_vec)

    # Initialise readings
    readings = np.zeros(len(sensors))

    for i, sensor in enumerate(sensors):
        # Get sensor normal in body frame
        n_vec = sensor.get_normal()
        
        # Compute angle between sun vector and sensor normal
        phi = np.arccos(np.dot(sun_vec, n_vec))

        # Check if angle is within the field of view
        if np.abs(phi) < max_FOV:
            # Compute S_rel
            readings[i] = np.cos(phi)
            if not exact:
                # Add noise to the readings
                readings[i] += np.random.normal(0, noise)
                # Ensure readings are non-negative
                readings[i] = max(0, readings[i])
        else:
            readings[i] = np.random.normal(0, noise)
            readings[i] = max(0, readings[i])

    if exact: 
        print(f"Expected readings for Sun vector {sun_vec}: {readings}")
    else:
        print(f"Noisy expected readings for Sun vector {sun_vec}: {readings}")
    return readings

# Calibrate sensor readings
def calibrate_measurements(measurements, max_measure):
    """Calibrates the sensor readings based on reading at 90deg"""
    readings = [m/max_measure for m in measurements]
    # Adjust if any are maximised
    for i in len(readings):
        if readings[i] > 1.0:
            print("Warning: S_rel > 1.0 detected!")
            readings[i] = 0
    return readings


if __name__ == "__main__":

    sensor1 = SunSensor(1, "Top", [0.02, 0.05, 0.2], 0, 0, 0)
    sensor2 = SunSensor(2, "Right", [0, 0.05, 0.11], 0, 90, 0)
    sensor3 = SunSensor(3, "Left", [0.1, 0.05, 0.11], 0, -90, 0)
    sensor4 = SunSensor(4, "Back", [0.05, 0.1, 0.11], 90, 0, 0)
    sensor5 = SunSensor(5, "Bottom", [0.02, 0.05, 0], 180, 0, 0)

    sensors = [sensor1, sensor2, sensor3, sensor4, sensor5]

    model_sun_sensors(sensors, [0.1, 0.1, 0.2])
