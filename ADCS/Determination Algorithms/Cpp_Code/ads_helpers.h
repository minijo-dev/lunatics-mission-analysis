// ekf_helpers.h

#ifndef EKF_HELPERS_H
#define EKF_HELPERS_H

#include <Eigen/Dense>
#include <tuple>

struct Constants {
    Eigen::Vector2d a;       // Not used in provided logic
    Eigen::Matrix3d J;
    Eigen::Matrix<double, 7, 7> Q;
    Eigen::Matrix<double, 6, 6> R;
    double sigma_tau;
    double sigma_q;
    double sigma_Acc;
    double sigma_Mag;
};

struct EKFResult {
    Eigen::VectorXd nts_state;
    Eigen::Matrix<double, 7, 7> nts_P;
};

Eigen::Vector3d eci_vector(double lat, double lon);
Eigen::Matrix3d inertial2body(const Eigen::VectorXd& state);
Eigen::Matrix3d body2inertial(const Eigen::VectorXd& state);
Eigen::Vector4d QUEST_simp(const Eigen::Vector3d& a_ref, const Eigen::Vector3d& m_ref,
    const Eigen::Vector3d& a_meas, const Eigen::Vector3d& m_meas,
    const Constants& Constant);

Eigen::Matrix3d adjoint(const Eigen::Matrix3d& M);

Eigen::Vector4d QUEST_full(const Eigen::Vector3d& a_known, const Eigen::Vector3d& m_known,
    const Eigen::Vector3d& a_sensor, const Eigen::Vector3d& m_sensor,
    const Constants& Constant);

Eigen::Matrix4d Omega_func(const Eigen::Vector3d& omega) ;
Eigen::Matrix<double, 4, 3> Xi_func(const Eigen::VectorXd& init_state);
Eigen::Matrix3d Pi_func(const Eigen::VectorXd& init_state, const Constants& Constant);


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

Eigen::VectorXd StateTransitionFunction(const Eigen::VectorXd& init_state, const Eigen::Vector3d& g_sensor,
double dt, const Constants& Constant);

Eigen::MatrixXd StateTransitionMatrix(const Eigen::VectorXd& init_state, const Constants& Constant);
Eigen::MatrixXd ProcessNoiseMatrix(double dt, const Constants& Constant);
Eigen::MatrixXd SensorJacobian(const Eigen::VectorXd& nts_state, const Eigen::Vector3d& sensor_vector);
Eigen::Matrix<double, 6, 7> MeasurementModelJacobian(const Eigen::Matrix<double, 6, 1>& h, const Eigen::VectorXd& nts_state);

#endif // EKF_HELPERS_H
