import numpy as np
# from scipy.optimize import least_squares
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


def find_sun_direction(sensors, readings, X0, max_iter=200, tol=1e-6, n_theta=100):
    """A function that estimates the Sun's direction vector using readings from multiple light sensors in WLS.
    Args:
        sensors (list): List of SunSensor objects.
        readings (np array): Array of sensor readings.
        X0 (np array): Initial guess for the Sun's position (in body frame).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        n_theta (int): Number of theta values to sample in the cone.
    Returns:
        X (np array): WLS estimation of the Sun's position (in body frame) [X, Y, Z].
    """

    def sensor_cone(sensor, S_rel):
        """Calculates the cone of potential Sun direction vectors based on sensor reading.
        Args:
            sensor (SunSensor object): The sensor object.
            S_rel (float): The reading from that sensor.
        Returns:
            u_cone (np array): Array of direction vectors in the body frame.
        """

        # Initialise cone array
        u_cone = []

        # Calculate half-angle of cone (phi)
        phi = np.arccos(S_rel)      # (radians)

        for theta in np.linspace(0, 2*np.pi, n_theta):
            u_sensor = np.array([
                np.sin(phi) * np.cos(theta),    # x-component
                np.sin(phi) * np.sin(theta),    # y-component
                np.cos(phi)                     # z-component
            ])

            # Normalise the vector
            u_sensor /= np.linalg.norm(u_sensor)

            # Transform from sensor frame to body frame
            u_body = sensor.transform_to_body(u_sensor)

            # Append to direction vector cone array
            u_cone.append(u_body)

        return np.array(u_cone)
    

    def residual_func(X, sensor, u_cone):
        """Calculates the residual (perpendicular distance) for the WLS function.
        Args:
            X (np array): Current estimate of Sun's direction vector in body frame.
            sensor (SunSensor object): The sensor object.
            u_cone (np array): Array of direction vectors in the body frame.
        Returns:
            residuals (np array): Perpendicular distance residual for each direction vector.
        """

        # Sensor position in body frame
        p_vec = sensor.pos_vec

        residuals = []
        for u in u_cone:
            cross_prod = np.cross((X - p_vec), u)
            residual = np.linalg.norm(cross_prod) / np.linalg.norm(u)

            # Append the residual
            residuals.append(residual)

        return np.array(residuals)
    

    def build_WLS_matrices(sensors, readings):
        """Build the H, W, and Y matrices for the WLS function."""

        # Number of sensors (should be 3)
        N = len(sensors)

        # Weighting matrix (W)
        W = np.diag(readings**2)
        W /= np.linalg.trace(W)

        # Cones for each sensor
        cones = []
        for i, sensor in enumerate(sensors):
            cones.append(sensor_cone(sensor, readings[i]))

        # Design matrix (H)
        H = []
        for i in range(N):
            for u in cones[i]:
                H.append(u)
        H = np.array(H)

        # Measurement vector (Y)
        Y = np.array([u for u_cone in cones for u in u_cone])
        


    # Select only three highest readings
    three_sensors = sorted(zip(sensors, readings), key=lambda x: x[1], reverse=True)[:3]
    sensors = [sensor for sensor, _ in three_sensors]
    readings = np.array([reading for _, reading in three_sensors])

    








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

    # Initial guess for Sun's position
    X = X0

    for iteration in range(max_iter):
        # Initialise residuals storage
        residuals = np.zeros((len(readings), 1))

        # Initialise Jacobian matrix
        H = np.zeros((len(sensors), 3))

        # Initialise weight matrix
        W = np.zeros((len(sensors), len(sensors)))

        # Iterate over each sensor and populate matrices
        for i, (sensor, S_rel) in enumerate(zip(sensors, readings)):
            # Sensor's local z-axis in body frame
            n_vec = sensor.z_in_body

            # Sensor position in body frame
            p_vec = sensor.pos_vec

            # Relative vector from sensor to current Sun estimate
            r_vec = X - p_vec

            # Weight for the sensor based on its reading
            W[i, i] = S_rel**2

            # Compute residuals for each sensor 
            lhs = np.dot(r_vec, n_vec)**2
            rhs = np.linalg.norm(r_vec)**2 * S_rel**2
            residuals[i] = lhs - rhs

            # Gradient of the residual w.r.t. x, y, z
            norm_r = np.linalg.norm(r_vec)
            grad_x = (2 * (np.dot(r_vec, n_vec) * r_vec[0] - norm_r**2 * n_vec[0])) / norm_r**3
            grad_y = (2 * (np.dot(r_vec, n_vec) * r_vec[1] - norm_r**2 * n_vec[1])) / norm_r**3
            grad_z = (2 * (np.dot(r_vec, n_vec) * r_vec[2] - norm_r**2 * n_vec[2])) / norm_r**3

            H[i, :] = [grad_x, grad_y, grad_z]

        
        # Solve the WLS system
        HTW = H.T @ W
        try: 
            dx = np.linalg.inv(HTW @ H) @ HTW @ residuals
        except np.linalg.LinAlgError:
            print("Singular matrix, skipping update...")
            break

        # Update Sun's position estimate
        X = X - dx.flatten()

        # Check for convergence
        if np.linalg.norm(dx) < tol:
            print(f"Convergence achieved after {iteration + 1} iterations.")
            break

    return X

def expected_readings(sensors, sun_pos, max_FOV=80):
    """
    Calculate expected readings for each sensor based on Sun's position.

    Args:
        sensors (list): List of SunSensor objects.
        sun_pos (np.array): Sun's position in body frame [X, Y, Z].
        max_FOv (float): Maximum field of view for the sensor (degrees).

    Returns:
        readings (np.array): Expected readings for the sensors
    """

    # Convert max_FOV to radians
    max_FOV = np.deg2rad(max_FOV)

    # Initialise readings array
    readings = np.zeros(len(sensors))

    for i, sensor in enumerate(sensors):
        # Get sensor position in body frame and local z-axis
        p_vec = sensor.pos_vec
        n_vec = sensor.z_in_body

        # Vector from sensor to Sun
        r_vec = sun_pos - p_vec

        # Compyte the angle between the sensor's z-axis and the Sun's direction
        cos_phi = np.dot(r_vec, n_vec) / np.linalg.norm(r_vec) / np.linalg.norm(n_vec)
        phi = np.arccos(cos_phi)

        # If the angle is within the sensor's FOV, calculate S_rel
        if abs(phi) < max_FOV:
            readings[i] = cos_phi
        else:
            readings[i] = 0

    return readings


if __name__ == "__main__":

    sensor1 = SunSensor(1, "Top", [0.05, 0.05, 0.2], 0, 0, 0)
    sensor2 = SunSensor(2, "Bottom", [0.05, 0.05, 0], 180, 0, 0)
    sensor3 = SunSensor(3, "Front", [0, 0.05, 0.1], 0, -90, 0)
    sensor4 = SunSensor(4, "Back", [0.1, 0.05, 0.1], 0, 90, 0)
    sensor5 = SunSensor(5, "Left", [0.05, 0, 0.1], 90, 0, 0)
    sensor6 = SunSensor(6, "Right", [0.5, 0.1, 0.1], -90, 0, 0)

    sensors = [sensor1, sensor2, sensor3, sensor4, sensor5]

    example_readings = expected_readings(sensors, np.array([1e6, 1e6, 1e6]))
    print(f"Expected readings: {example_readings}")

    readings = np.array([0.8, 0.2, 0.9, 0.4, 0.6])  # Examples

    find_sun_direction(sensors, readings, np.array([1, 0, 0]))

    # init_guess = np.array([10, 10, 10])
    # sun_pos = WLS(sensors, example_readings, init_guess)
    # print(f"Estimated Sun Position: {sun_pos}")