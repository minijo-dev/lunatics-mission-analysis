// ekf_helpers.cpp

#include "ads_helpers.h"
#include <cmath>
#include <iostream>

using namespace Eigen;

Vector3d eci_vector(double lat, double lon) {
    return Vector3d(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
}

Matrix3d inertial2body(const VectorXd& state) {
    Vector4d q = state.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    Matrix3d C;
    C << 1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
         2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1),
         2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2);
    
    return C;
}

Matrix3d body2inertial(const VectorXd& state) {
    return inertial2body(state).transpose();
}

// QUEST Functions //
Vector4d QUEST_simp(const Vector3d& a_ref, const Vector3d& m_ref,
               const Vector3d& a_meas, const Vector3d& m_meas,
               const Constants& Constant) {
    Matrix3d W;
    W.col(0) = a_meas.cross(a_ref);
    W.col(1) = m_meas.cross(m_ref);
    W.col(2) = a_meas.cross(a_ref) + m_meas.cross(m_ref);

    Vector3d v = W.col(2);
    double sigma = W.trace();
    Matrix3d Z = W - W.transpose();
    Vector3d z(Z(1, 2), Z(2, 0), Z(0, 1));

    double normv = v.norm();
    Vector4d q;
    q << sqrt((sigma + normv) / 2),
         z(0) / (2 * sqrt((sigma + normv) / 2)),
         z(1) / (2 * sqrt((sigma + normv) / 2)),
         z(2) / (2 * sqrt((sigma + normv) / 2));
    return q;
}


// Helper function: computes adjoint of a 3x3 matrix
Matrix3d adjoint(const Matrix3d& M) {
    Matrix3d adj;
    adj(0,0) = M(1,1)*M(2,2) - M(1,2)*M(2,1);
    adj(0,1) = M(0,2)*M(2,1) - M(0,1)*M(2,2);
    adj(0,2) = M(0,1)*M(1,2) - M(0,2)*M(1,1);
    adj(1,0) = M(1,2)*M(2,0) - M(1,0)*M(2,2);
    adj(1,1) = M(0,0)*M(2,2) - M(0,2)*M(2,0);
    adj(1,2) = M(0,2)*M(1,0) - M(0,0)*M(1,2);
    adj(2,0) = M(1,0)*M(2,1) - M(1,1)*M(2,0);
    adj(2,1) = M(0,1)*M(2,0) - M(0,0)*M(2,1);
    adj(2,2) = M(0,0)*M(1,1) - M(0,1)*M(1,0);
    return adj;
}

Vector4d QUEST_full(const Vector3d& a_known, const Vector3d& m_known,
               const Vector3d& a_sensor, const Vector3d& m_sensor,
               const Constants& Constant) {
    // Build attitude profile matrix B
    Matrix3d B = Constant.a[0] * a_sensor * a_known.transpose() + Constant.a[1] * m_sensor * m_known.transpose();

    // Compute helper matrices and vectors
    double sigma = B.trace();
    Matrix3d S = B + B.transpose();
    Vector3d Z = Constant.a[0] * a_sensor.cross(a_known) + Constant.a[1] * m_sensor.cross(m_known);

    // Calculate characteristic equation parameters
    double nu = 0.5 * S.trace();
    double k = (adjoint(S)).trace();
    double Delta = B.determinant();

    double e = nu*nu - k;
    double f = nu*nu + Z.dot(Z);
    double g = Delta + Z.transpose() * S * Z;
    double h = Z.transpose() * S * S * Z;

    // Newton-Raphson method to find largest eigenvalue lambda
    double lambda = sigma + Z.norm();  // initial guess
    double tol = 1e-7;
    double error = 1.0;
    int max_iter = 1000;
    int iter = 0;

    while (std::abs(error) > tol && iter < max_iter) {
        // Characteristic equation value and derivative
        double CE = std::pow(lambda, 4) - (e + f) * std::pow(lambda, 2) - g * lambda + (e*f + g*nu - h);
        double dCE = 4 * std::pow(lambda, 3) - 2 * (e + f) * lambda - g;

        double lambda_new = lambda - CE / dCE;
        error = lambda_new - lambda;
        lambda = lambda_new;
        iter++;
    }

    if (iter == max_iter) {
        std::cerr << "Warning: QUEST lambda solver did not converge\n";
    }

    // Compute quaternion components
    double alpha = lambda*lambda - sigma*sigma + k;
    double beta = lambda - sigma;
    Vector3d X = (alpha * Matrix3d::Identity() + beta * S + S * S) * Z;
    double gamma = (lambda + sigma) * alpha - Delta;

    // Form quaternion and normalize
    Vector4d q;
    q << gamma, X(0), X(1), X(2);
    q.normalize();

    return q;
}
////

//EKF Helpers Functions//

Matrix4d Omega_func(const Vector3d& omega) {
    Matrix4d Omega;
    Omega <<  0,     -omega(0), -omega(1), -omega(2),
              omega(0), 0,       omega(2), -omega(1),
              omega(1), -omega(2), 0,       omega(0),
              omega(2), omega(1), -omega(0), 0;
    
    return Omega;

}

Eigen::Matrix<double, 4, 3> Xi_func(const Eigen::VectorXd& init_state) {
    Vector4d q = init_state.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    Eigen::Matrix<double, 4, 3> Xi;
    Xi << -q1, -q2, -q3,
          q0, -q3, q2,
          q3, q0, -q1,
          -q2, q1, q0 ;

    return Xi;
}

Eigen::Matrix3d Pi_func(const Eigen::VectorXd& init_state, const Constants& Constant){

    Matrix3d J = Constant.J;
    double J1 = J(0, 0), J2 = J(1, 1), J3 = J(2, 2);

    double J23 = J2 - J3;
    double J31 = J3 - J1;
    double J12 = J1 - J2;

    Vector3d w = init_state.tail<3>();
    double w1 = w(0), w2 = w(1), w3 = w(2);

    Eigen::Matrix3d Pi;
    Pi << 0,            (J23 / J1) * w3,  (J23 / J1) * w2,
          (J31 / J2) * w3,  0,            (J31 / J2) * w1,
          (J12 / J3) * w2,  (J12 / J3) * w1,  0;

    
    return Pi;
}



VectorXd StateTransitionFunction(const VectorXd& init_state, const Vector3d& g_sensor,
                                 double dt, const Constants& Constant) {
    Vector4d q = init_state.head<4>();
    Vector3d w = init_state.tail<3>();
    Matrix3d J = Constant.J;

    Matrix4d Omega = Omega_func(g_sensor);

    Vector4d dq = 0.5 * Omega * q;
    Vector4d q_new = q + dq * dt;
    q_new.normalize();

    Vector3d torque = Constant.sigma_tau * Vector3d::Ones();  // σ_tau * ones(3,1)
    Vector3d dw = -J.inverse() * (torque - w.cross(J * w));
    Vector3d w_new = w + dw * dt;

    VectorXd state_new(7);
    state_new.head<4>() = q_new;
    state_new.tail<3>() = w_new;
   
    return state_new;
}


MatrixXd StateTransitionMatrix(const VectorXd& init_state, const Vector3d& g_sensor,const Constants& Constant) {
    // Vector3d w = init_state.tail<3>();
    Matrix4d Omega = Omega_func(g_sensor);
    Eigen::Matrix<double, 4, 3> Xi = Xi_func(init_state);
    Matrix3d Pi = Pi_func(init_state, Constant);
    Eigen::Matrix<double, 3, 4> zeroes = Eigen::Matrix<double, 3, 4>::Zero();
    Eigen::Matrix<double, 7, 7> F;
    F << 0.5*Omega, 0.5*Xi,
         zeroes, Pi;
    
    return F;
}

MatrixXd ProcessNoiseMatrix(double dt, const Constants& Constant) {
    
    Eigen::Matrix4d Qq = Constant.sigma_q * Constant.sigma_q * Eigen::Matrix4d::Identity();

    // Qw = sigma_tau^2 * J^-1 * (J^-1)^T * dt
    Eigen::Matrix3d J_inv = Constant.J.inverse();
    Eigen::Matrix3d Qw = Constant.sigma_tau * Constant.sigma_tau * J_inv * J_inv.transpose() * dt;

    // Construct the full 7x7 matrix
    Eigen::Matrix<double, 7, 7> Q = Eigen::Matrix<double, 7, 7>::Zero();
    Q.topLeftCorner<4, 4>() = Qq;
    Q.bottomRightCorner<3, 3>() = Qw;

    return Q;
}

MatrixXd SensorJacobian(const VectorXd& nts_state, const Vector3d& sensor_vector) {
  
    double x = sensor_vector(0);
    double y = sensor_vector(1);
    double z = sensor_vector(2);
 
    double q0 = nts_state(0);
    double q1 = nts_state(1);
    double q2 = nts_state(2);
    double q3 = nts_state(3);

    Eigen::Matrix<double, 3, 4> H_sensor;
    H_sensor << 
        2 * (x*q0 + y*q3 - z*q2),  2 * (x*q1 + y*q2 + z*q3),  2 * (-x*q2 + y*q1 - z*q0),  2 * (-x*q3 + y*q0 + z*q1),
        2 * (-x*q3 + y*q0 + z*q1), 2 * (x*q2 - y*q1 + z*q0),  2 * (x*q1 + y*q2 + z*q3),   2 * (-x*q0 - y*q3 + z*q2),
        2 * (x*q2 - y*q1 + z*q0),  2 * (x*q3 - y*q0 - z*q1),  2 * (x*q0 + y*q3 - z*q2),   2 * (x*q1 + y*q2 + z*q3);

    return H_sensor;
    
}

Eigen::Matrix<double, 6, 7> MeasurementModelJacobian(const Eigen::Matrix<double, 6, 1>& h, const VectorXd& nts_state) {
    Eigen::Vector3d h_acc = h.segment<3>(0); // h(0:2)
    Eigen::Vector3d h_mag = h.segment<3>(3); // h(3:5)
    
    Eigen::Matrix<double, 3, 4> H_acc = SensorJacobian(nts_state, h_acc);
    Eigen::Matrix<double, 3, 4> H_mag = SensorJacobian(nts_state, h_mag);
    
    // Build the full 6x7 H matrix
    Eigen::Matrix<double, 6, 7> H = Eigen::Matrix<double, 6, 7>::Zero();
    H.block<3, 4>(0, 0) = H_acc;
    H.block<3, 4>(3, 0) = H_mag;

    return H;
}


//EKF//

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
) {
    EKFResult result;

    // === PREDICT ===
    Eigen::VectorXd nts_state = StateTransitionFunction(init_state, g_sensor, dt, Constant); // 7x1
    Eigen::Matrix<double, 7, 7> F = StateTransitionMatrix(init_state, g_sensor, Constant);
    Eigen::Matrix<double, 7, 7> Q = ProcessNoiseMatrix(dt, Constant);
    Eigen::Matrix<double, 7, 7> nts_P = F * init_P * F.transpose() + Q;

    // === UPDATE ===
    Eigen::Matrix<double, 6, 1> z;
    z << a_sensor, m_sensor;

    Eigen::Matrix3d C = inertial2body(nts_state);
    Eigen::Vector3d a_expected = C * a_known;
    Eigen::Vector3d m_expected = C * m_known;

    Eigen::Matrix<double, 6, 1> h;
    h << a_expected, m_expected;
    Eigen::Matrix<double, 6, 7> H = MeasurementModelJacobian(h, nts_state);
    Eigen::Matrix<double, 6, 1> v = z - h;
    Eigen::Matrix<double, 6, 6> S = H * nts_P * H.transpose() + Constant.R;

    Eigen::Matrix<double, 7, 6> K = nts_P * H.transpose() * S.inverse();

    // Correction
    nts_state = nts_state + K * v;
    nts_P = (Eigen::Matrix<double, 7, 7>::Identity() - K * H) * nts_P;

    // Normalize quaternion
    nts_state.segment<4>(0).normalize();

    // Save results
    result.nts_state = nts_state;
    result.nts_P = nts_P;

    return result;
}
////"
