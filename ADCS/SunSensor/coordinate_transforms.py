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

    def transform_to_sensor(self, V):
        """Transform a vector from the body frame to the sensor frame.
        Args:
            V (np array): The vector to be transformed, in the body frame, V = [X,Y,Z]
        Returns:
            v (np array): The vector transformed into the local sensor frame, v = [x,y,z]
        """
        V = np.array(V)
        v = self.rotmat_SB @ (V + self.pos_vec)
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
        V = self.rotmat_BS @ v - self.pos_vec
        print(f"Vector in Sensor {self.number} ({self.face}) frame: {v}")
        print(f"Vector in Body frame: {V}")
        return V        


if __name__ == "__main__":

    sensor1 = SunSensor(1, "Top", [0.5, 0.5, 2], 0, 0, 0)
    sensor2 = SunSensor(2, "Bottom", [0.5, 0.5, 0], 180, 0, 0)
    sensor3 = SunSensor(3, "Front", [0, 0.5, 1], 0, -90, 0)
    sensor4 = SunSensor(4, "Back", [1, 0.5, 1], 0, 90, 0)
    sensor5 = SunSensor(5, "Left", [0.5, 0, 1], 90, 0, 0)
    sensor6 = SunSensor(6, "Right", [0.5, 1, 1], -90, 0, 0)

    sensor5.transform_to_body([0, 0, 1])