import numpy as np
from sun_sensors import SunSensor
from scipy.optimize import least_squares


def sun_triangulation(sensors, readings):
    """
    Triangulates the Sun position based on Sun Sensor readings.
    
    Args:
        sensors (list): List of SunSensor objects.
        readings (np array): Array of sensor readings.
    """

    def residuals(X, three_sensors, three_readings):
        """
        Residual function for least squares optimisation.

        Args:
            X (np array): The parameters to optimise (coordinates of the Sun).
            three_sensors (list): The three SunSensor objects with maximum readings.
            three_readings (np array): The readings from the three sensors.

        Returns:
            residuals (np array): Array of residuals.
        """

        residuals = []

        for sensor, S_rel in zip(three_sensors, three_readings):
            # Extract sensor's body-frame z-axis (unit vec along local sensor z-axis)
            n_vec = sensor.rotmat_BS @ np.array([0, 0, 1])

            # Position vector of sensor in body frame
            p = sensor.pos_vec

            # Vector from sensor to proposed sun position
            r_vec = X - p

            # Cone equation:
            # ((X - p) . n_vec)^2 = S_rel^2 * ||X - p||^2
            lhs = (np.dot(r_vec, n_vec))**2
            rhs = S_rel**2 * np.linalg.norm(r_vec)**2

            # Calculate residual
            residuals.append(lhs - rhs)

        return residuals
    
    # Find the three sensors with maximum readings
    max_idx = np.argsort(readings)[-3:]
    three_sensors = [sensors[i] for i in max_idx]
    three_readings = readings[max_idx]

    # Initial guess for the Sun's position
    X0 = np.array([10, 10, 10])

    # Solve the system using least squares:
    # SHOULD PROBABLY CHANGE TO USE WLS OR NLLS OR SOMETHING
    result = least_squares(residuals, X0, args=(three_sensors, three_readings))

    sun_position = result.x

    print(f"Triangulated Sun poisition in body frame: {sun_position}")



def WLS(sensors, readings, X0, max_iter=200, tol=1e-6):
    """
    Performs Weighted Least Squares triangulation for Sun position.
    
    Args:
        sensors (list): List of SunSensor objects.
        readings (np array): Array of sensor readings.
        X0 (np array): Initial guess for the Sun's position (in body frame).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        X (np array): WLS estimation of the Sun's position (in body frame) [X, Y, Z].
    """

    # Initial guess for the Sun's position
    X = X0

    for iteration in range(max_iter):
        # Initialise residuals storage
        residuals = np.zeros((len(readings), 1))

        # Initialise Jacobian matrix
        H = np.zeros((len(sensors), 3))

        # Initialise weight matrix
        W = np.zeros((len(sensors), len(sensors)))

        for i, (sensor, S_rel) in enumerate(zip(sensors, readings)):
            # Sensor's local z-axis in body frame
            n_vec = sensor.rotmat_BS @ np.array([0, 0, 1])

            # Sensor position in body frame
            p = sensor.pos_vec

            # Relative vector from sensor to current Sun estimate
            r_vec = X - p

            # Weight for the sensor based on its reading
            W[i, i] = S_rel**2      # Square of cosine (S_rel) as weight

            # Gradient of the residual w.r.t. x, y, z
            norm_r = np.linalg.norm(r_vec)
            grad_x = (2 * (np.dot(r_vec, n_vec) * r_vec[0] - norm_r**2 * n_vec[0])) / norm_r**3
            grad_y = (2 * (np.dot(r_vec, n_vec) * r_vec[1] - norm_r**2 * n_vec[1])) / norm_r**3
            grad_z = (2 * (np.dot(r_vec, n_vec) * r_vec[2] - norm_r**2 * n_vec[2])) / norm_r**3

            H[i, :] = [grad_x, grad_y, grad_z]

            # # Residual for sensor i
            # lhs = np.dot(r_vec, n_vec)**2
            # rhs = np.linalg.norm(r_vec)**2 * S_rel**2
            # residuals[i] = lhs - rhs

            
        # Solve the WLS system
        HTW = H.T @ W
        try: 
            dx = np.linalg.inv(HTW @ H) @ HTW @ residuals
        except np.linalg.LinAlgError:
            print("Singular matrix, skipping update...")
            break

        # Update Sun's position estimate
        X = X - dx
        
        # Check for convergence
        if np.linalg.norm(dx) < tol:
            print(f"Convergence achieved after {iteration + 1} iterations.")
            break

    return X
    
    


if __name__ == "__main__":
    
    # Define sensor positions and rotation matrices
    sensor1 = SunSensor(1, "Top", [0.5, 0.5, 2], 0, 0, 0)
    sensor2 = SunSensor(2, "Bottom", [0.5, 0.5, 0], 180, 0, 0)
    sensor3 = SunSensor(3, "Front", [0, 0.5, 1], 0, -90, 0)
    sensor4 = SunSensor(4, "Back", [1, 0.5, 1], 0, 90, 0)
    sensor5 = SunSensor(5, "Left", [0.5, 0, 1], 90, 0, 0)
    # sensor6 = SunSensor(6, "Right", [0.5, 1, 1], -90, 0, 0)

    # sensors = [sensor1, sensor2, sensor3, sensor4, sensor5, sensor6]
    sensors = [sensor1, sensor2, sensor3, sensor4, sensor5]

    # Test readings from each of the sensors, S_rel
    # readings = np.array([0.76, 0.1, 0.5, 0.4, 0.84, 0.2])
    readings = np.array([0.76, 0.1, 0.5, 0.4, 0.84])

    # sun_triangulation(sensors, readings)
    sun_position = WLS(sensors, readings, np.array([10, 10, 10]))
    print(f"Estimated Sun position: {sun_position}")