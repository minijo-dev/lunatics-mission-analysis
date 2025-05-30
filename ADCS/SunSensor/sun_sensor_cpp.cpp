#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;
using namespace Eigen;

#define DEG2RAD(x) ((x) * M_PI / 180.0)

class SunSensor {
public:
    int number;
    string face;
    Vector3d pos_vec;
    Matrix3d rotmat_SB;
    Matrix3d rotmat_BS;
    Vector3d z_in_body;

    SunSensor(int number, string face, vector<double> position_vector, double phi, double theta, double psi)
        : number(number), face(face)
    {
        pos_vec = Vector3d(position_vector[0], position_vector[1], position_vector[2]);

        phi = DEG2RAD(phi);
        theta = DEG2RAD(theta);
        psi = DEG2RAD(psi);

        rotmat_SB << 
            cos(theta)*cos(psi), -cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta),
            cos(theta)*sin(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi),
            -sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta);

        rotmat_BS = rotmat_SB.inverse();
        z_in_body = rotmat_BS * Vector3d(0, 0, 1);
    }

    Vector3d transform_to_sensor(Vector3d V) {
        Vector3d v = rotmat_SB * V;
        cout << "Vector in Body frame: " << V.transpose() << endl;
        cout << "Vector in Sensor " << number << " (" << face << ") frame: " << v.transpose() << endl;
        return v;
    }

    Vector3d transform_to_body(Vector3d v) {
        Vector3d V = rotmat_BS * v;
        cout << "Vector in Sensor " << number << " (" << face << ") frame: " << v.transpose() << endl;
        cout << "Vector in Body frame: " << V.transpose() << endl;
        return V;
    }

    Vector3d get_normal() {
        return z_in_body;
    }
};

pair<Vector3d, vector<int>> find_sun(
    vector<SunSensor>& sensors, 
    VectorXd readings, 
    Vector3d X0, 
    double Srel_tol = 0.1, 
    int max_iter = 1000, 
    double tol = 1e-6, 
    double learning_rate = 0.1) 
{
    vector<int> idx(readings.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&readings](int i1, int i2) { return readings(i1) < readings(i2); });

    vector<SunSensor> selected_sensors = {sensors[idx[2]], sensors[idx[1]], sensors[idx[0]]};
    Vector3d selected_readings(readings(idx[2]), readings(idx[1]), readings(idx[0]));

    if ((selected_readings.array() < Srel_tol).any()) {
        cout << "Not enough valid readings from sensors." << endl;
        return {Vector3d::Zero(), {0, 0, 0}};
    }

    vector<int> selected_indices = {idx[2] + 1, idx[1] + 1, idx[0] + 1};

    Matrix3d H;
    for (int i = 0; i < 3; ++i) {
        H.row(i) = selected_sensors[i].get_normal().transpose();
    }

    Matrix3d W = selected_readings.array().square().matrix().asDiagonal();
    W /= W.trace();

    Vector3d Y = selected_readings;
    Vector3d Xhat = X0.normalized();

    for (int iter = 0; iter < max_iter; ++iter) {
        Vector3d r = Y - H * Xhat;
        Vector3d gradient = -2.0 * H.transpose() * W * r;
        Vector3d dX = learning_rate * gradient;

        if (dX.cwiseAbs().sum() <= tol) break;

        Xhat -= dX;
        Xhat.normalize();

        // cout << "Estimated Sun direction vector: " << Xhat.transpose() << endl;
    }

    return {Xhat, selected_indices};
}

// Compute expected sensor readings given a sun vector
VectorXd expected_readings(
    const vector<SunSensor>& sensors,
    const Vector3d& sun_vec,
    bool exact = true,
    double noise = 0.01,
    double max_FOV_deg = 80.0)
{
    const double max_FOV = max_FOV_deg * M_PI / 180.0;  // Convert FOV to radians
    Vector3d sun_dir = sun_vec.normalized();            // Ensure sun_vec is unit vector
    VectorXd readings(sensors.size());
    readings.setZero();

    // Random number generator for noise
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, noise);

    for (size_t i = 0; i < sensors.size(); ++i) {
        Vector3d n_vec = sensors[i].get_normal();
        double phi = acos(n_vec.dot(sun_dir));

        if (abs(phi) < max_FOV) {
            readings(i) = cos(phi);
            if (!exact) {
                readings(i) += d(gen);
                readings(i) = max(0.0, readings(i));
            }
        } else {
            readings(i) = exact ? 0.0 : max(0.0, d(gen));
        }
    }

    cout << (exact ? "Expected" : "Noisy expected")
         << " readings for Sun vector " << sun_dir.transpose()
         << ": " << readings.transpose() << endl;

    return readings;
}

int main() {
    vector<SunSensor> sensors = {
        SunSensor(1, "Top",    {0.02, 0.05, 0.2}, 0,   0,   0),
        SunSensor(2, "Right",  {0.00, 0.05, 0.11}, 0,  90,  0),
        SunSensor(3, "Left",   {0.10, 0.05, 0.11}, 0, -90,  0),
        SunSensor(4, "Back",   {0.05, 0.10, 0.11}, 90,  0,  0),
        SunSensor(5, "Bottom", {0.02, 0.05, 0.00},180,  0,  0)
    };

    
    Vector3d sun_guess(0.1, 0.1, 0.2);
    VectorXd readings(5);
    readings << 0.6, 0.5, 0.1, 0.4, 0.3;

    Vector3d

    auto [sun_direction, used_sensors] = find_sun(sensors, readings, sun_guess);

    cout << "Final estimated sun direction: " << sun_direction.transpose() << endl;
    cout << "Used sensors: ";
    for (int s : used_sensors) cout << s << " ";
    cout << endl;

    return 0;
}