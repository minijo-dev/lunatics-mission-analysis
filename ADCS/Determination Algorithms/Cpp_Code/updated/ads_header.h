// ekf_helpers.h

#ifndef EKF_HELPERS_H
#define EKF_HELPERS_H

#include <Eigen/Dense>
#include <tuple>

struct Constants {
    Eigen::Vector2d a;       
    Eigen::Matrix3d J;
    double g;
    Eigen::Matrix<double, 7, 7> Q;
    Eigen::Matrix<double, 6, 6> R;
    double sigma_Qq;
    double sigma_Qb;
    double sigma_Acc;
    double sigma_Mag;
};

struct EKFResult {
    Eigen::VectorXd state;
    Eigen::Matrix<double, 7, 7> P;
    Eigen::Vector3d w;
    Eigen::Vector3d euler;
};

Eigen::Vector3d eci_vector(double lat, double lon);
Eigen::Matrix3d inertial2body(const Eigen::VectorXd& state);

Eigen::Matrix3d adjoint(const Eigen::Matrix3d& M);

Eigen::Vector4d QUEST(const Eigen::Vector3d& a_known, const Eigen::Vector3d& m_known,
    const Eigen::Vector3d& a_sensor, const Eigen::Vector3d& m_sensor,
    const Constants& Constant);

Eigen::Matrix<double, 4, 3> Xi_func(const Eigen::VectorXd& init_state);

EKFResult ekf(
    Eigen::VectorXd& init_state,           // 7x1
    const Eigen::Matrix<double, 7, 7>& init_P,
    double mag_dec_meas,
    double prop_time,
    double dt,
    bool new_meas,
    const Constants& Constant
);

Eigen::VectorXd StateTransitionFunction(const Eigen::VectorXd& init_state, const Constants& Constant);
Eigen::MatrixXd StateTransitionMatrix(const Eigen::VectorXd& init_state);
Eigen::MatrixXd SensorJacobian(const Eigen::VectorXd& init_state, const double mag_dec_meas, const Constants& Constant);
Eigen::Vector3d Acc2Body(const Eigen::VectorXd& init_state, const Constants& Constant);
Eigen::Vector3d Mag2Body(const Eigen::VectorXd& init_state, const double mag_dec_meas);
Eigen::Vector3d q2e(const Eigen::VectorXd& init_state);
#endif 
