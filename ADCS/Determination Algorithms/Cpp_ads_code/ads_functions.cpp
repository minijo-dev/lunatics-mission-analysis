#include "ads_helpers.h"
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace Eigen;

Eigen::Vector3d randomVector(double min = -0.05, float max = 0.05) {
    // Static so it's initialized only once and reused every call
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    return Eigen::Vector3d(dist(gen), dist(gen), dist(gen));
}

Eigen::Vector3d eci_vector(double lat, double lon) {
    return Eigen::Vector3d(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
}

Eigen::Matrix3d inertial2body(const Eigen::VectorXd& state) {
    Eigen::Vector4d q = state.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    Eigen::Matrix3d C;
    C << 1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
         2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1),
         2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2);
    
    return C;
}

// Helper function: computes adjoint of a 3x3 matrix
Eigen::Matrix3d adjoint(const Eigen::Matrix3d& M) {
    Eigen::Matrix3d adj;
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

// QUEST // 
Eigen::Vector4d QUEST(const Eigen::Vector3d& a_known, const Eigen::Vector3d& m_known, 
    const Eigen::Vector3d& a_sensor, const Eigen::Vector3d& m_sensor,const Constants& Constant) {
    // Build attitude profile matrix B
    Eigen::Matrix3d B = Constant.a[0] * a_sensor * a_known.transpose() + Constant.a[1] * m_sensor * m_known.transpose();

    // Compute helper matrices and vectors
    double sigma = B.trace();
    Eigen::Matrix3d S = B + B.transpose();
    Eigen::Vector3d Z = Constant.a[0] * a_sensor.cross(a_known) + Constant.a[1] * m_sensor.cross(m_known);

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
    Eigen::Vector3d X = (alpha * Eigen::Matrix3d::Identity() + beta * S + S * S) * Z;
    double gamma = (lambda + sigma) * alpha - Delta;

    // Form quaternion and normalize
    Eigen::Vector4d q;
    q << gamma, X(0), X(1), X(2);
    q.normalize();

    return q;
}
/////////////////


// EKF Helper Functions //
Eigen::Matrix<double, 4, 3> Xi_func(const Eigen::VectorXd& init_state) {
    Eigen::Vector4d quat = init_state.head<4>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);

    Eigen::Matrix<double, 4, 3> Xi;
    Xi << -q1, -q2, -q3,
          q0, -q3, q2,
          q3, q0, -q1,
          -q2, q1, q0 ;

    return Xi;
}

Eigen::VectorXd StateTransitionFunction(const Eigen::VectorXd& init_state, const Constants& Constant) {
    Eigen::Vector3d b = init_state.tail<3>();
    
    ////////////////////////////////////// TO CHANGE ONCE VALUES ARRIVE /////////////////////////////////////////////////////////////////
    // gyro_meas = getGyro();
    Eigen::Vector3d v1 = randomVector();
    Eigen::Vector3d gyro_meas(1.57, 0, 0); // filler for now
    gyro_meas = gyro_meas + v1;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    Eigen::Vector3d corr_gyro = gyro_meas - b;
    Eigen::Matrix<double, 4, 3> Xi = Xi_func(init_state);
    Eigen::VectorXd qdot = 0.5*Xi*corr_gyro;
    
    Eigen::VectorXd f(7);
    f.setZero();
    f.head<4>() = qdot;
    return f;
}

Eigen::MatrixXd StateTransitionMatrix(const Eigen::VectorXd& init_state) { 
    Eigen::Vector4d quat = init_state.head<4>();
    Eigen::Vector3d b = init_state.tail<3>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);


    ////////////////////////////////////// TO CHANGE ONCE VALUES ARRIVE /////////////////////////////////////////////////////////////////
    // gyro_meas = getGyro();
    Eigen::Vector3d v1 = randomVector();
    Eigen::Vector3d gyro_meas(1.57, 0, 0); // filler for now
    gyro_meas = gyro_meas + v1;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    

    Eigen::Vector3d corr_gyro = gyro_meas - b;
    float p = corr_gyro(0), q = corr_gyro(1), r = corr_gyro(2);

    Eigen::Matrix< double, 7, 7> F; 
    
    F << 0, -0.5*p, -0.5*q, -0.5*r, 0.5*q1, 0.5*q2, 0.5*q3, 
        0.5*p, 0, 0.5*r, -0.5*q, -0.5*q0, 0.5*q3, -0.5*q2,
        0.5*q, -0.5*r, 0, 0.5*p, -0.5*q3, -0.5*q0, 0.5*q1,
        0.5*r, 0.5*q, -0.5*p, 0, 0.5*q2, -0.5*q1, -0.5*q0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0;

    return F;
}

MatrixXd SensorJacobian(const VectorXd& init_state, const double mag_dec_meas, const Constants& Constant) {
    Eigen::Vector4d quat = init_state.head<4>();
    Eigen::Vector3d b = init_state.tail<3>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);

    double smag = sin(mag_dec_meas);
    double cmag = cos(mag_dec_meas);

    double g = Constant.g;

    Eigen::Matrix< double, 6, 7> H;
    H << 2 * g * q2, -2 * g * q3,  2 * g * q0, -2 * g * q1, 0, 0, 0,
        -2 * g * q1, -2 * g * q0, -2 * g * q3, -2 * g * q2, 0, 0, 0,
        0, 4 * g * q1, 4 * g * q2, 0, 0, 0, 0,
        2 * q3 * smag, 2 * q2 * smag, 2 * q1 * smag - 4 * q2 * cmag, 2 * q0 * smag - 4 * q3 * cmag, 0, 0, 0,
        -2 * q3 * cmag, 2 * q2 * cmag - 4 * q1 * smag, 2 * q1 * cmag, -2 * q0 * cmag - 4 * q3 * smag, 0, 0, 0,
        2 * q2 * cmag - 2 * q1 * smag, 2 * q3 * cmag - 2 * q0 * smag, 2 * q0 * cmag + 2 * q3 * smag, 2 * q1 * cmag + 2 * q2 * smag, 0, 0, 0;

    return H;
}

Eigen::Vector3d Acc2Body(const VectorXd& init_state, const Constants& Constant){
    Eigen::Vector4d quat = init_state.head<4>();
    Eigen::Vector3d b = init_state.tail<3>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);

    double g = Constant.g;

    Eigen::Vector3d acc_est;
    acc_est<< -2 * g * (q1 * q3 - q2 * q0),
        - 2 * g * (q2 * q3 + q1 * q0),
        - g * (1 - 2 * (q1 * q1 + q2 * q2));
    return acc_est;
}

Eigen::Vector3d Mag2Body(const VectorXd& init_state, const double mag_dec_meas){
    Eigen::Vector4d quat = init_state.head<4>();
    Eigen::Vector3d b = init_state.tail<3>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);

    double smag = sin(mag_dec_meas);
    double cmag = cos(mag_dec_meas);

    Eigen::Vector3d mag_est;
    mag_est << smag * (2 * q0 * q3 + 2 * q1 * q2) - cmag * (2 * q2 * q2 + 2 * q3 * q3 - 1), 
        -cmag * (2 * q0 * q3 - 2 * q1 * q2) - smag * (2 * q1 * q1 + 2 * q3 * q3 - 1), 
         cmag * (2 * q0 * q2 + 2 * q1 * q3) - smag * (2 * q0 * q1 - 2 * q2 * q3);

    return mag_est;
}

Eigen::Vector3d q2e(const VectorXd& init_state){
    Eigen::Vector4d quat = init_state.head<4>();
    Eigen::Vector3d b = init_state.tail<3>();
    float q0 = quat(0), q1= quat(1), q2 = quat(2), q3 = quat(3);

    double pi = std::numbers::pi;

    Eigen::Vector3d euler;
    euler << atan2(2 * (q0 * q1 + q2 * q3), q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2) * 180 / pi,
             asin (2 * (q0 * q2 - q1 * q3)) * 180 / pi,
             atan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * 180 / pi;
    
    return euler;
}

/////////////////


// EKF Function //

EKFResult ekf(
    Eigen::VectorXd& init_state,           // 7x1
    Eigen::MatrixXd& init_P,
    double mag_dec_meas,
    double prop_time,
    double timesteps_ekf,
    bool new_meas,
    const Constants& Constant
) {
    EKFResult result;

    // Predict // 
    for (int count = 0; count < timesteps_ekf; ++count){
        Eigen::VectorXd f = StateTransitionFunction(init_state, Constant);
        init_state = init_state + (prop_time/timesteps_ekf)*f;
        init_state.segment<4>(0).normalize();

        Eigen::Matrix< double, 7, 7> F = StateTransitionMatrix(init_state);
        init_P = init_P + prop_time * (F * init_P + init_P * F.transpose() + Constant.Q);

        // run interrupts
        if (new_meas){

            // update code

            Eigen::Vector3d acc_est = Acc2Body(init_state, Constant);
            Eigen::Vector3d mag_est = Mag2Body(init_state, mag_dec_meas);

            Eigen::VectorXd z(6);
            z.head<3>() = acc_est;
            z.tail<3>() = mag_est;
            Eigen::Matrix< double, 6, 7> H = SensorJacobian(init_state, mag_dec_meas, Constant);

            Eigen::Matrix<double, 6, 6> S = H * init_P * H.transpose() + Constant.R;
            Eigen::Matrix<double, 7, 6> K = init_P * H.transpose() * S.inverse();
            
            result.P = (Eigen::Matrix<double, 7, 7>::Identity() - K * H) * init_P;

            ///////////////////////////// TO CHANGE WHEN NEW VALUES ARRIVE //////////////////////////////////////////////////
            Eigen::Vector3d v1 = randomVector();
            Eigen::Vector3d v2 = randomVector();
    
            
            Eigen::Vector3d acc_meas(0, 0, -9.81);
            Eigen::Vector3d mag_meas(1, 0, 0);

            acc_meas = acc_meas + v1;
            mag_meas = mag_meas + v2;
            // acc_meas = getAcc()
            // mag_meas = getMag()
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            Eigen::VectorXd meas(6);
            meas.head<3>() = acc_meas;
            meas.tail<3>() = mag_meas;

            result.state = init_state + K*(meas-z);
            result.state.segment<4>(0).normalize();

            ////////////////////////////////////// TO CHANGE ONCE VALUES ARRIVE /////////////////////////////////////////////////////////////////
            // gyro_meas = getGyro();
            Eigen::Vector3d v3 = randomVector();
            Eigen::Vector3d gyro_meas(1.57, 0, 0); // filler for now
            gyro_meas = gyro_meas + v3;

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            

            result.w = gyro_meas - result.state.tail<3>();
            result.euler = q2e(result.state);

            return result;
            break;
        }

        ////////////////////////////////////// TO CHANGE ONCE VALUES ARRIVE /////////////////////////////////////////////////////////////////
        // gyro_meas = getGyro();
        Eigen::Vector3d v1 = randomVector();
        Eigen::Vector3d gyro_meas(1.57, 0, 0); // filler for now
        gyro_meas = gyro_meas + v1;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
        
        result.state = init_state;
        result.P  = init_P;
        result.w = gyro_meas - result.state.tail<3>();
        result.euler = q2e(result.state);


    }
    return result;


}
