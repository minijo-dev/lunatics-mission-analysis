
// To compile atm
//g++ -std=c++20 ads_integrated2.cpp ads_functions.cpp -I /usr/include/eigen3 -o ads_program2


#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include "ads_helpers.h"
#include <fstream>
#include <numbers>

using namespace Eigen;
using namespace std;


int main() {

    // CSV file setup //
    std::ofstream file_state("log_state.csv");
    std::ofstream file_euler("log_euler.csv");
    std::ofstream file_angv("log_angv.csv");

    if (!file_state.is_open()) {
        std:cout << "Can't open file_state \n";
    }

    file_state << "q0 q1 q2 q3 bx by bz\n";
    file_euler << "roll pitch yaw\n";
    file_angv << "wx wy wz\n";

    // Constants //
    Constants Constant;
    Constant.a << (Vector2d() << 0.5, 0.5).finished();
    Constant.J << 0.0017, 0, 0,
                    0, 0.0015, 0,
                    0, 0, 0.0022;
    Constant.g = 9.81;
    
    Constant.sigma_Qq = 0.00005;
    Constant.sigma_Qb = 0.000001;
    Constant.Q.setZero(7, 7);
    Constant.Q.block<4, 4>(0, 0) = pow(Constant.sigma_Qq, 2) * Matrix4d::Identity();
    Constant.Q.block<3, 3>(4, 4) = pow(Constant.sigma_Qb, 2) * Matrix3d::Identity();

    Constant.sigma_Acc = 0.05;
    Constant.sigma_Mag = 0.02;
    Constant.R.setZero(6, 6);
    Constant.R.block<3, 3>(0, 0) = pow(Constant.sigma_Acc, 2) * Matrix3d::Identity();
    Constant.R.block<3, 3>(3, 3) = pow(Constant.sigma_Mag, 2) * Matrix3d::Identity();

    // Initialisation
    int test_time = 500; 
    double timesteps = 10000; 
    bool initialise = true; ///CAN BECHANGED
    double prop_time = 0.01; ///CAN BECHANGED
    double timesteps_ekf = 1; ///CAN BE CHANGED

    // Declaring variables
    VectorXd quat(4);
    VectorXd init_state(7);
    VectorXd state(7);
    MatrixXd init_P;
    MatrixXd P;

    init_P.setZero(7, 7);
    P.setZero(7, 7);

    

    // Known vectors //
    double latitude = -33.88893628858434 * M_PI / 180.0;
    double longitutde = 151.19446182380247 * M_PI / 180.0;

    Vector3d R_ECI = eci_vector(latitude, longitutde);
    Vector3d a_known = 9.81 * R_ECI.normalized();  // known acceleration in the lab
    Vector3d m_known(0, 0, 1); // known magnetic field in the lab
    double mag_dec_meas = 0;
    
    
    ////////////////////////////////// TO BE CHANGED WHEN INTERRUPTS ARE ADDED ////////////////////////////////////
    bool new_meas = true; 
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // Main loop //
    for (int count = 0; count < timesteps; ++count) {

        // Initialisation function //
        if (initialise){

            /////////////////// CHANGE WHEN ADDING IN LIVE MEASUREMENTS /////////////////////////////////
            // acc_meas = getAcc();
            // mag_meas = getMag();
            Eigen::Vector3d acc_meas(0,0,-9.81);
            Eigen::Vector3d mag_meas(1,0,0);
            //////////////////////////////////////////////////////////////////////////////////////////////

            quat = QUEST(a_known, m_known, acc_meas, mag_meas, Constant);
            init_state(0) = 1;
            // init_state.head<4>() = quat;
            init_P.block<4, 4>(0, 0) = pow(0.001, 2) * Matrix4d::Identity();
            init_P.block<3, 3>(3, 3) = pow(0.0001, 2) * Matrix3d::Identity();
            std::cout << "QUEST function executed";
            initialise = false;
        }
        else {
            init_state = state;
            init_P = P;
        }

        // EKF //
        EKFResult ekf_result = ekf(init_state, init_P, mag_dec_meas, prop_time, timesteps_ekf, new_meas, Constant);
        
        file_state << ekf_result.state.transpose();
        file_state << "\n";

        file_euler << ekf_result.euler.transpose();
        file_euler << "\n";

        file_angv << ekf_result.w.transpose();
        file_angv << "\n";

        state = ekf_result.state;
        P = ekf_result.P;

        

    }





}
