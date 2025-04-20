
%% Initial values
x0 = [1,0,0,0,0,0,0.01];
% x0 = [0,0,0.6,0.8,0.25,0.3,-0.2];
P0 = eye(7);

x_true0 = [0, 0, 0.6, 0.8, 0.25, 0.3, -0.2];

%% Simulation timing and Setup
Ts = 300; % sample time (sec)
n = 1000; % number of time steps
dt = Ts/n; % change in time
time = linspace(0,Ts,n);

% initialising arrays
x = zeros(7,n);
x(:,1) = x0;

x_true = zeros(7,n);
x_true(:,1) = x_true0;

% known sun vector (inertial frame)
s_known = [1,0,0];

%% Constants
% j = [0.004433; 0.004433; 0.01922]; 
j = [0.0017, 0.0015, 0.0022];
J = diag(j);

tau = [1e-9; 1.5e-9; 1e-8]; % external torque disturbances (3x1)

% noise from gyro
w_noise = [0.01, 0.02, 0.01];
% noise from sun sensor
s_noise = [0.01, 0.01, 0.01];


%% Items to tune
% model noise (7x7)
sigmaQ = 0.1;
Q = eye(7)*sigmaQ^2;

% sensor noise (5x5)
sigmaGyro = 0.001;
sigmaSun = 0.5;
R = eye(6);
R(1:3,:) = R(1:3,:) * sigmaGyro^2;
R(4:6,:) = R(4:6,:) * sigmaSun^2;


H_gyro = [zeros(3,4) eye(3)];



for i = 1:n
    %% Propagating true values
    omega_true = small_omega(x_true(1,i), x_true(2,i), x_true(3,i), x_true(4,i));
    w_true = [x_true(5,i), x_true(6,i), x_true(7,i)];
    q_true = [x_true(1,i), x_true(2,i), x_true(3,i), x_true(4,i)];
    rot = inert2body(q_true);
    s_true = rot*s_known.';

    x_true(:,i+1) = StateTransitionFunction(x_true(:,i), dt, omega_true, w_true, J, [0;0;0]);

    x_true(1:4, i+1) = x_true(1:4, i+1) / norm(x_true(1:4, i+1));



    %% Input values
    % gyro measurements - testing only
    w_measured = w_true + w_noise;

    % sun sensor measurements - testing only
    s_measured = s_true.' + s_noise;
   
    %% Predict
    omega = small_omega(x(1,i), x(2,i), x(3,i), x(4,i));
    w = [x(5,i), x(6,i), x(7,i)];
    x(:,i+1) = StateTransitionFunction(x(:,i), dt, omega, w, J, tau); % initial / previous state in here
    
    Omega = big_omega(w);
    Pi = pi4F(j, w);
    F = StateTransitionMatrix(omega, Omega, Pi);
    P = F*P0*F.' + Q ;


    %% Update

    H_sun = compute_H_sun(x(1:4,i+1),s_known); % (3x4)
    H = [H_gyro;
         H_sun eye(3)]; % (6x7)
    S = H*P*H.' + R + 1e-6*eye(6); % adding regularisation...
    K = P*H.' / S; 

    s_hat = inert2body(x(1:4,i+1)) * s_known.';

    z = [w_measured s_measured];
    h = [x(5:7,i+1).' s_hat.'];

    x(:,i+1) = x(:,i+1) + K*(z.'-h.');
    P = (eye(7) - K*H)*P;

    %% Normalise quaternions
    x(1:4, i+1) = x(1:4, i+1) / norm(x(1:4, i+1));

    P0 = P;
end


%% Graphing Results

% figure(1)
% plot(time, wx_true, "Color", "red")
% hold on
% plot(time, x(5,1:1000), "Color","blue")
% 
% figure(2)
% plot(time, wy_true, "Color", "red")
% hold on
% plot(time, x(6,1:1000), "Color","blue")
% 
% figure(3)
% plot(time, wz_true, "Color", "red")
% hold on
% plot(time, x(7,1:1000), "Color","blue")
% 
% figure(4)
% plot(x(1,1:500), "Color","red")
% hold on
% plot(x(2,1:500), "Color","blue")
% hold on
% plot(x(3,1:500), "Color","green")
% hold on
% plot(x(4,1:500), "Color","black")

figure(5)
plot(time, x_true(4,1:1000), "Color","red")
hold on
plot(time,x(4,1:1000), "Color","blue" )

figure(6)
plot(time, x_true(7,1:1000), "Color","red")
hold on
plot(time,x(7,1:1000), "Color","blue" )



%% FUNCTIONS %%

% Euler to Quaternion
function q_vec = e2q(e_vec)
% e_vec (1x3)
% q_vec (1x4)
    q0 = cosd(e_vec(3)/2)*cosd(e_vec(2)/2)*cosd(e_vec(1)/2) + sind(e_vec(3)/2)*sind(e_vec(2)/2)*sind(e_vec(1)/2);
    q1 = cosd(e_vec(3)/2)*cosd(e_vec(2)/2)*sind(e_vec(1)/2) - sind(e_vec(3)/2)*sind(e_vec(2)/2)*cosd(e_vec(1)/2);
    q2 = cosd(e_vec(3)/2)*sind(e_vec(2)/2)*cosd(e_vec(1)/2) + sind(e_vec(3)/2)*cosd(e_vec(2)/2)*sind(e_vec(1)/2);
    q3 = -cosd(e_vec(3)/2)*sind(e_vec(2)/2)*sind(e_vec(1)/2) + sind(e_vec(3)/2)*cosd(e_vec(2)/2)*cosd(e_vec(1)/2);
    q_vec = [q0 q1 q2 q3];
end


function omega = small_omega(q0,q1,q2,q3)
    omega = [-q1 -q2 -q3;
              q0 -q3 q2;
              q3 q0 -q1;
              -q2 q1 q0];
end


% state transition function x_k+1
function x_kp1 = StateTransitionFunction(x, dt, omega, w, J, tau)
    f = [0.5*omega*w.'; -J \ (tau - cross(w.', J*w.'))];
    x_kp1 = x + f*dt;
end

function Omega = big_omega(w)
% matrix helps define qdot (4x4)
Omega = [0 -w(1) -w(2) -w(3);
         w(1) 0 w(3) -w(2);
         w(2) -w(3) 0 w(1);
         w(3) w(2) -w(1) 0];
end

function Pi = pi4F(j, w)
    % dwdot/dw (3x3)
    j23 = j(2) - j(3);
    j31 = j(3) - j(1);
    j12 = j(1) - j(2);
    Pi = [0             j23*w(3)/j(1)      j23*w(2)/j(1);
          j31*w(3)/j(2) 0                  j31*w(1)/j(2);
          j12*w(2)/j(3)   j12*w(1)/j(3)        0];
end

% state transition matrix / Jacobian (7x7)
function F = StateTransitionMatrix(omega, Omega, Pi)
    F = [0.5*Omega 0.5*omega;
        zeros(3,4) Pi];
end

% rotation matrix from quaternion to DCM, inertial to body frame
function R = inert2body(q)
    q0 = q(1); 
    q1 = q(2); 
    q2 = q(3); 
    q3 = q(4);
    l1= q0^2 + q1^2 - q2^2 - q3^2;
    l2 = 2*(q1*q2 + q0*q3);
    l3 = 2*(q1*q3 - q0*q2);
    m1 = 2*(q1*q2 - q0*q3);
    m2 = q0^2 -q1^2 + q2^2 - q3^2;
    m3 = 2*(q2*q3 +q0*q1);
    n1 = 2*(q0*q2 + q1*q3);
    n2 = 2*(q2*q3 - q0*q1);
    n3 = q0^2 -q1^2 - q2^2 +q3^2;

    R = [l1 l2 l3;
         m1 m2 m3;
         n1 n2 n3];
end

% Jacobian for observation matrix - sun sensor portion
function H_sun = compute_H_sun(q, s_known)
% s_known is in inertial frame

    q0 = q(1); 
    q1 = q(2); 
    q2 = q(3); 
    q3 = q(4);
    sx = s_known(1); 
    sy = s_known(2); 
    sz = s_known(3);

    % Partial derivatives (d(sb)/dq) for each component of sun vector
    dR1_dq = [ 2*q0*sx + 2*q3*sy - 2*q2*sz;
               2*q1*sx + 2*q2*sy + 2*q3*sz;
              -2*q2*sx + 2*q1*sy + 2*q0*sz;
              -2*q3*sx - 2*q0*sy + 2*q1*sz ];

    dR2_dq = [-2*q3*sx + 2*q0*sy + 2*q1*sz;
              2*q2*sx - 2*q1*sy + 2*q0*sz;
              2*q1*sx + 2*q2*sy + 2*q3*sz;
             -2*q0*sx - 2*q3*sy + 2*q2*sz ];

    dR3_dq = [ 2*q2*sx - 2*q1*sy + 2*q0*sz;
              2*q3*sx + 2*q0*sy - 2*q1*sz;
              2*q0*sx - 2*q3*sy - 2*q2*sz;
              2*q1*sx + 2*q2*sy + 2*q3*sz ];

    % Stack into a 3x4 Jacobian
    H_sun = [dR1_dq.'; dR2_dq.'; dR3_dq.'];
end
