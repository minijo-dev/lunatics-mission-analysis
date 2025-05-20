// ekf_helpers.h

#ifndef EKF_HELPERS_H
#define EKF_HELPERS_H

#include <Eigen/Dense>
#include <tuple>

struct Constants {
    Eigen::Vector2d a;       // Not used in provided logic
    Eigen::Matrix3d J;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    double sigma_tau;
    double sigma_q;
    double sigma_Acc;
    double sigma_Mag;
};

Eigen::Vector3d eci_vector(double lat, double lon);
Eigen::Matrix3d inertial2body(const Eigen::VectorXd& state);
Eigen::Matrix3d body2inertial(const Eigen::VectorXd& state);
Eigen::Vector4d QUEST_simp(const Vector3d& a_ref, const Vector3d& m_ref,
    const Vector3d& a_meas, const Vector3d& m_meas,
    const Constants& Constant);

Matrix3d adjoint(const Matrix3d& M);

Vector4d QUEST_full(const Vector3d& a_known, const Vector3d& m_known,
    const Vector3d& a_sensor, const Vector3d& m_sensor,
    const Constants& Constant);

Matrix4d Omega_func(const Vector3d& omega) ;
Eigen::Matrix<double, 4, 3> Xi_func(const std::vector<double>& init_state);
Eigen::Matrix3d Pi_func(const std::vector<double>& init_state, const Constants& Constant);


EKFResult ekf(
    const Eigen::VectorXd& init_state,           // 7x1
    const Eigen::Matrix<double, 7, 7>& init_P,
    const Eigen::Vector3d& a_known,
    const Eigen::Vector3d& m_known,
    const Eigen::Vector3d& a_sensor,
    const Eigen::Vector3d& m_sensor,
    const Eigen::Vector3d& g_sensor,
    double dt,
    const Constants& Constant
);

VectorXd StateTransitionFunction(const VectorXd& init_state, const Vector3d& g_sensor,
double dt, const Constants& Constant);

MatrixXd StateTransitionMatrix(const VectorXd& init_state, const Constants& Constant);
MatrixXd ProcessNoiseMatrix(double dt, const Constants& Constant);
MatrixXd SensorJacobian(const VectorXd& nts_state, const Vector3d& sensor_vector);
Eigen::Matrix<double, 6, 7> MeasurementModelJacobian(const Eigen::Matrix<double, 6, 1>& h, const VectorXd& nts_state);

#endif // EKF_HELPERS_H
