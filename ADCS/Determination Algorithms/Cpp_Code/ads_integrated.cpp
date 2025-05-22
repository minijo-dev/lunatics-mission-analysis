
// To compile atm
//g++ ads_integrated.cpp ads_helpers.cpp -I /usr/include/eigen3 -o ads_program



#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include "ads_helpers.h"
#include <fstream>
using namespace Eigen;
using namespace std;

int main() {
    // CSV file setup
    std::ofstream file("log.csv");
    std::ofstream file_true("log_true.csv");

    if (!file.is_open()) {
        std:cout << "Can't open file \n";
    }

    file << "q0 q1 q2 q3 wx wy wz\n";
    file_true << "q0 q1 q2 q3 wx wy wz\n";

    // Setup Values - Variable //
    double lat = -33.88893628858434 * M_PI / 180.0;
    double lon = 151.19446182380247 * M_PI / 180.0;
    double alt = 19.8;
    ////

    Vector3d R_ECI = eci_vector(lat, lon);
    Vector3d a_known = 9.81 * R_ECI.normalized();  // known acceleration at site

    Vector3d m_known(0, 0, 1); // known magnetic field (dummy)

    // Constants
    Constants Constant;
    Constant.a << (Vector2d() << 0.5, 0.5).finished();
    Constant.J << 0.0017, 0, 0,
                    0, 0.0015, 0,
                    0, 0, 0.0022;;
    Constant.Q = MatrixXd::Identity(7, 7) * pow(0.01, 2);
    Constant.sigma_tau = 1e-7;
    Constant.sigma_q = 1e-8;
    Constant.sigma_Acc = 0.01;
    Constant.sigma_Mag = 0.01;

    Constant.R.setZero(6, 6);
    Constant.R.block<3, 3>(0, 0) = pow(Constant.sigma_Acc, 2) * Matrix3d::Identity();
    Constant.R.block<3, 3>(3, 3) = pow(Constant.sigma_Mag, 2) * Matrix3d::Identity();

    // Initialisation
    int count = 0;

    // Time Setup - Variable //
    int iter = 10000;
    double dt = 0.1;
    ////

    // Starting States - Variable //
    VectorXd init_state_true(7);
    init_state_true << 0, 0, 0.6, 0.8, 0.05, -0.1, 0.15;

    VectorXd nts_state(7);
    nts_state << 1, 0, 0, 0, 0.1, 0.1, 0.1;
    MatrixXd nts_P = MatrixXd::Identity(7, 7);
    ////

    // Declaring variables
    VectorXd init_state(7);
    MatrixXd init_P(7, 7);

    // QUEST Version - Variable //
    bool simp = false;
    ////

    // std::vector<double> q0_plot{1};
    // std::vector<double> q0_true_plot{0};
    // std::vector<double> time{0};

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 0.001);

    for (int count = 0; count < iter; ++count) {
        // True state propagation
        VectorXd nts_state_true = StateTransitionFunction(init_state_true, init_state_true.segment<3>(4), dt, Constant);
        nts_state_true.head<4>().normalize();
        
        Matrix3d C = inertial2body(nts_state_true);

        // Fake Sensor Inputs//
        Vector3d a_sensor = C * a_known + Vector3d(noise(generator), noise(generator), noise(generator));
        Vector3d m_sensor = C * m_known + Vector3d(noise(generator), noise(generator), noise(generator));
        Vector3d g_sensor = nts_state_true.tail<3>() + Vector3d(noise(generator), noise(generator), noise(generator));
        ////

        // q0_true_plot.push_back(nts_state_true(2)); // q2

        init_state_true = nts_state_true;


        Eigen::Vector4d quest_quat;

        if (count == 0) {
            if (simp == true) {
                Vector4d quest_quat = QUEST_simp(a_known, m_known, a_sensor, m_sensor, Constant);
                std::cout << "Should never see me\n";
            } else {
                Vector4d quest_quat = QUEST_full(a_known, m_known, a_sensor, m_sensor, Constant);
            init_state.head<4>() = quest_quat;
            init_state.tail<3>() = nts_state.tail<3>();
            init_P = MatrixXd::Identity(7, 7);
            std::cout << "QUEST executed\n";}
        } else {
            init_state = nts_state;
            init_P = nts_P;
        }

        EKFResult ekf_result = ekf(init_state, init_P, a_known, m_known, a_sensor, m_sensor, g_sensor, dt, Constant);
        nts_state = ekf_result.nts_state;
        nts_P = ekf_result.nts_P;

        VectorXd error = nts_state_true - nts_state;

        file << nts_state.transpose();
        file << "\n";

        file_true << nts_state_true.transpose();
        file_true << "\n";
        
        // q0_plot.push_back(nts_state(2)); // q2
        // time.push_back(time.back() + dt);
    }

    file.close();
    file_true.close();
    std::cout << "file logged";
    return 0;
}
