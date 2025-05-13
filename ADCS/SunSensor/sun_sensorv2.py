"""
AERO4701 Space Engineering 3
Lunar Atmospheric Investigations with CubeSats (LUNATICS)

A class and functions to simulate a sun sensor system on a CubeSat.
"""

import numpy as np

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



def find_sun(sensors, readings, X0, max_iter=1000, tol=1e-6, learning_rate=0.1):
    """"""

    # Select top three sensors based on readings
    sorted_idx = np.argsort(readings)[-3:]
    sensors = [sensors[i] for i in sorted_idx]
    readings = readings[sorted_idx]

    # Check three sensors actually have values
    if readings[0] < 0.1 or readings[1] < 0.1 or readings[2] < 0.1:
        print("Not enough valid readings from sensors.")
        return None

    # Design matrix (H) [3x3]
    H = np.array([sensor.get_normal() for sensor in sensors])

    # Weight matrix (W) [3x3]
    W = np.diag(readings**2)
    W /= np.trace(W)  # Normalise

    # Observation vector (Y) [3x1]
    Y = readings

    # Initial sun direction vector estimate
    Xhat = X0 / np.linalg.norm(X0)    # Normalise

    # Weighted least squares estimation
    iter_count = 0
    for iter in range(max_iter):
        iter_count += 1
        # Compute residuals
        residuals = Y - np.dot(H, Xhat)

        # Compute gradient
        gradient = -2 * np.dot(H.T, W @ residuals)

        # Direction update
        dX = learning_rate * gradient

        # Check if update is significant enough to continue
        if np.sum(np.abs(dX)) > tol:
            # Update direction vector
            Xhat -= dX
            Xhat /= np.linalg.norm(Xhat)   # Normalise
        else:
            break

    print(f"Estimated sun direction vector: {Xhat} (after {iter_count} iterations)")

    return Xhat


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
    # print(f"-------angle: {np.rad2deg(np.arccos(readings))}")

    return readings
    
def pointing_accuracy(v_true, v_est):
    """Compute the pointing accuracy of the resulting sun direction vector."""
    # Find angle between true and estimated vectors
    dot_product = np.dot(v_true, v_est)
    true_mag = np.linalg.norm(v_true)
    est_mag = np.linalg.norm(v_est)
    angle = np.arccos(dot_product / (true_mag * est_mag))
    # Convert to degrees
    angle = np.rad2deg(angle)
    print(f"Pointing accuracy: {angle} degrees")
    return angle


if __name__ == "__main__":

    sensor1 = SunSensor(1, "Top", [0.05, 0.05, 0.2], 0, 0, 0)
    sensor2 = SunSensor(2, "Bottom", [0.05, 0.05, 0], 180, 0, 0)
    sensor3 = SunSensor(3, "Front", [0, 0.05, 0.1], 0, -90, 0)
    sensor4 = SunSensor(4, "Back", [0.1, 0.05, 0.1], 0, 90, 0)
    sensor5 = SunSensor(5, "Left", [0.05, 0, 0.1], 90, 0, 0)
    sensor6 = SunSensor(6, "Right", [0.5, 0.1, 0.1], -90, 0, 0)


    sensors = [sensor1, sensor2, sensor3, sensor4, sensor5]

    test_dir = np.array([1, 1, 1])

    example_readings = expected_readings(sensors, test_dir)
    clean_est = find_sun(sensors, example_readings, np.array([1, 0, 0]))

    noisy_readings = expected_readings(sensors, test_dir, exact=False)
    noise_est = find_sun(sensors, noisy_readings, np.array([1, 0, 0]))

    pointing_accuracy(test_dir, clean_est)
    pointing_accuracy(test_dir, noise_est)