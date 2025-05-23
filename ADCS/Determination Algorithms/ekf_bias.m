
clear all; close all;
Constant = ConstantClass(); 
Constant.g = 9.81;
% Constant.sigma_q = 0.001;
% Constant.sigma_bias = 0.0001;
% Constant.P = [Constant.sigma_q^2 * eye(3) zeros(3,3);
%               zeros(3,3)                  Constant.sigma_bias^2];
Constant.sigma_Qq = 0.00005;
Constant.sigma_Qb = 0.000001;
Constant.Q = [Constant.sigma_Qq^2 * eye(4) zeros(4,3);
              zeros(3,4)                   Constant.sigma_Qb^2 * eye(3)];
Constant.sigma_Acc = 0.05;
Constant.sigma_Mag = 0.02;
Constant.R = [Constant.sigma_Acc^2 * eye(3) zeros(3,3);
              zeros(3,3)                    Constant.sigma_Mag^2* eye(3)];


% gyro_meas = [0; 0; 0];
% acc_meas = [0; 0; -9.81];
% mag_meas = [1; 0; 0];
% mag_dec_meas = 0;
% prop_time = 0.01;
% timesteps = 1;

init_state = [1;0;0;0;0;0;0];
% init_P = 0.001^2*eye(7);
init_P = diag([[1 1 1 1] * 0.001, [1 1 1] * 0.0001] .^ 2);

roll = zeros(100,1);
pitch= zeros(100,1);
yaw= zeros(100,1);
state = zeros(100,7);
w = zeros(100,3);

% [state, P, w, roll, pitch, yaw] = ekf(init_state, init_P, gyro_meas, acc_meas, mag_meas, mag_dec_meas, prop_time, timesteps, Constant);

gyro_meas = [1.57; 0; 0];  % Roll rate only
acc_meas = [0; 0; -9.81];  % Still level (not moving yet)
mag_meas = [1; 0; 0];
mag_dec_meas = 0;
prop_time = 0.01;
timesteps = 1;
% 
% 
% 
for i = 1:1000
    [state(i,:), P, w(i,:), roll(i), pitch(i), yaw(i)] = ekf(init_state, init_P, gyro_meas, acc_meas, mag_meas, mag_dec_meas, prop_time, timesteps, Constant);
    init_state = state(i,:).';
    init_P = P;
end
figure(1)
hold on
plot(roll)
plot(pitch)
plot(yaw)
hold off

figure(2)
hold on
plot(state(:,1))
plot(state(:,2))
plot(state(:,3))
plot(state(:,4))
hold off

figure(3)
hold on 
plot(w)
hold off

figure(4)
hold on
plot(state(:,5))
plot(state(:,6))
plot(state(:,7))


function [state, P, w, roll, pitch, yaw] = ekf(init_state, init_P, gyro_meas, acc_meas, mag_meas, mag_dec_meas, prop_time, timesteps, Constant)

    g = Constant.g;
    dt = prop_time/timesteps;

    gyro_meas = gyro_meas + 0.01*randn(3,1);
    acc_meas = acc_meas + 0.01*randn(3,1);
    mag_meas = mag_meas + 0.01*randn(3,1);
    mag_meas = unit_vector(mag_meas);

    %% Predict 
    % done in small increments to reduce linearisation error
    for i = 1:timesteps

        f = StateTransitionFunction(gyro_meas, init_state);
        init_state = init_state + dt*f;
        % init_state(1:4) = unit_vector(init_state(1:4));

    end

    init_state(1:4) = unit_vector(init_state(1:4));
    F = StateTransitionMatrix(gyro_meas, init_state);
    init_P = init_P + prop_time* (F*init_P + init_P*F'+ Constant.Q);

    %% Update
    % changing sensor measurements to body frame
    mag_est = Mag2Body(mag_dec_meas, init_state);
    acc_est = Acc2Body(init_state, g);

    z = [acc_est;
         mag_est];

    H = SensorJacobian(init_state, mag_dec_meas, g);

    % Kalman gain
    K = init_P * H' / (H* init_P * H' + Constant.R);

    % S = H * init_P * H' + Constant.R;
    % S = S + 1e-6 * eye(size(S));  % Regularization
    % K = init_P * H' / S;

    % S = H * init_P * H' + Constant.R;
    % K = init_P * H' * inv(S + 1e-8*eye(6));

    P = (eye(length(init_state)) - K*H) * init_P;% + 1e-9 * eye(7);

    meas = [acc_meas;
            mag_meas];
    state = init_state + K*(meas -z);

    state(1:4) = unit_vector(state(1:4));
    w = gyro_meas - state(5:7);

    [roll, pitch, yaw] = q2e(state(1:4));
  


end

function unit = unit_vector(vector)
    unit = vector/norm(vector);
end

function Xi = Xi_func(init_state)

    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    Xi = [-q1 -q2 -q3;
           q0 -q3 q2;
           q3 q0 -q1;
          -q2 q1 q0];

end

function corr_gyro = CorrectedGyro(gyro_meas, init_state)

    p = gyro_meas(1);
    q = gyro_meas(2);
    r = gyro_meas(3);

    bp = init_state(5);
    bq = init_state(6);
    br = init_state(7);


    corr_gyro = [p - bp;
                 q - bq;
                 r - br];
 
end

function f = StateTransitionFunction(gyro_meas, init_state)
   
    corr_gyro = CorrectedGyro(gyro_meas, init_state);
    Xi = Xi_func(init_state);
    qdot = 0.5*Xi*corr_gyro;

    f = [qdot;
         0;
         0;
         0];
end

function F = StateTransitionMatrix(gyro_meas, init_state)

    corr_gyro = CorrectedGyro(gyro_meas, init_state);

    p = corr_gyro(1);
    q = corr_gyro(2);
    r = corr_gyro(3);

    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    % checked matrix is correct
    F = [0, -0.5*p, -0.5*q, -0.5*r, 0.5*q1, 0.5*q2, 0.5*q3; 
         0.5*p, 0, 0.5*r, -0.5*q, -0.5*q0, 0.5*q3, -0.5*q2; 
         0.5*q, -0.5*r, 0, 0.5*p, -0.5*q3, -0.5*q0, 0.5*q1; 
         0.5*r, 0.5*q, -0.5*p, 0, 0.5*q2, -0.5*q1, -0.5*q0; 
         0, 0, 0, 0, 0, 0, 0; 
         0, 0, 0, 0, 0, 0, 0; 
         0, 0, 0, 0, 0, 0, 0];

end

function mag_est = Mag2Body(mag_dec_meas, init_state)
    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    smag = sin(mag_dec_meas);
    cmag = cos(mag_dec_meas);

    mag_est = [ smag * (2 * q0 * q3 + 2 * q1 * q2) - cmag * (2 * q2 * q2 + 2 * q3 * q3 - 1); 
               -cmag * (2 * q0 * q3 - 2 * q1 * q2) - smag * (2 * q1 * q1 + 2 * q3 * q3 - 1); 
                cmag * (2 * q0 * q2 + 2 * q1 * q3) - smag * (2 * q0 * q1 - 2 * q2 * q3)];
end

function acc_est = Acc2Body(init_state, g)

    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    acc_est = [-2 * g * (q1 * q3 - q2 * q0);
               - 2 * g * (q2 * q3 + q1 * q0);
               - g * (1 - 2 * (q1 * q1 + q2 * q2))];
    % R = inertial2body(init_state);
    % acc_est = R*[0; 0; -g];

end

function C = inertial2body(nts_state)

    q0 = nts_state(1);
    q1 = nts_state(2);
    q2 = nts_state(3);
    q3 = nts_state(4);

    l1 = q0^2 + q1^2 - q2^2 - q3^2;
    l2 = 2*(q1*q2 - q0*q3);
    l3 = 2*(q1*q3 + q0*q2);
    m1 = 2*(q1*q2 + q0*q3);
    m2 = q0^2 - q1^2 + q2^2 - q3^2;
    m3 = 2*(q2*q3 - q0*q1);
    n1 = 2*(q1*q3 - q0*q2);
    n2 = 2*(q0*q1 + q2*q3);
    n3 = q0^2 - q1^2 - q2^2 + q3^2;

    C = [l1 l2 l3;
         m1 m2 m3;
         n1 n2 n3];

end

function H = SensorJacobian(init_state, mag_dec_meas, g)

    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    smag = sin(mag_dec_meas);
    cmag = cos(mag_dec_meas);

    H = [ 2 * g * q2, -2 * g * q3,  2 * g * q0, -2 * g * q1, 0, 0, 0;
         -2 * g * q1, -2 * g * q0, -2 * g * q3, -2 * g * q2, 0, 0, 0;
          0, 4 * g * q1, 4 * g * q2, 0, 0, 0, 0;
          2 * q3 * smag, 2 * q2 * smag, 2 * q1 * smag - 4 * q2 * cmag, 2 * q0 * smag - 4 * q3 * cmag, 0, 0, 0; 
         -2 * q3 * cmag, 2 * q2 * cmag - 4 * q1 * smag, 2 * q1 * cmag, -2 * q0 * cmag - 4 * q3 * smag, 0, 0, 0; 
          2 * q2 * cmag - 2 * q1 * smag, 2 * q3 * cmag - 2 * q0 * smag, 2 * q0 * cmag + 2 * q3 * smag, 2 * q1 * cmag + 2 * q2 * smag, 0, 0, 0];
 
end

function [roll, pitch, yaw] = q2e(quat)

    q0 = quat(1);
    q1 = quat(2);
    q2 = quat(3);
    q3 = quat(4);

    roll = atan2(2 * (q0 * q1 + q2 * q3), q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2) * 180 / pi;
    pitch = asin (2 * (q0 * q2 - q1 * q3)) * 180 / pi;
    yaw = atan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * 180 / pi;

end
