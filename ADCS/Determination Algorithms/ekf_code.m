
% initial values
x0 = [1,0,0,0,0.5,0.5,0.5];
% x0 = [0,0,0.6,0.8,0.25,0.3,-0.2];
P0 = eye(7);

% Simulation timing
Ts = 500; % sample time (sec)
n = 1000; % number of time steps
dt = Ts/n; % change in time
time = linspace(0,Ts,n);

% constants
% j = [0.004433; 0.004433; 0.01922]; 
j = [0.0017, 0.0015, 0.0022];
J = diag(j);

tau = [1e-9; 1.5e-9; 1e-8]; % external torque disturbances (3x1)

% initialising arrays
angle = linspace(0,2*pi,n);
phi = angle;
theta = angle;
psi = linspace(0,pi/4,n);

wx_true = sin(phi);
wy_true = cos(theta);
wz_true = sin(psi);

x = zeros(7,n);
x(:,1) = x0;

euler_noise = randn(1,3);
% w_noise = randn(1,3);
w_noise = [0.01, 0.02, 0.01];
% w_noise = zeros(1,3);

% model noise (7x7)
sigmaQ = 0.001;
Q = eye(7)*sigmaQ^2;

% sensor noise (5x5)
sigmaR = 0.0001;
R = eye(3)*sigmaR^2;

H = [zeros(3,4) eye(3)];



for i = 1:n

    %% Input values
    % true euler angle (1x3)
    % euler_true = [phi(i) theta(i) psi(i)];
    % euler_measured = euler_true + euler_noise;
    % % measured quaternion (1x4) 
    % quat_measured = e2q(euler_measured);

    % true angular velocity (1x3)
    w_true = [wx_true(i) wy_true(i) wz_true(i)];
    w_measured = w_true + w_noise;

    % state
    % x = [quat_measured w_measured];
    
    %% Normalise quaternions
    % x(1:4, i) = x(1:4, i) / norm(x(1:4, i));

    %% Predict
    omega = small_omega(x(1,i), x(2,i), x(3,i), x(4,i));
    w = [x(5,i), x(6,i), x(7,i)];
    x(:,i+1) = StateTransitionFunction(x(:,i), dt, omega, w, J, tau); % initial / previous state in here
    
    Omega = big_omega(w);
    Pi = pi4F(j, w);
    F = StateTransitionMatrix(omega, Omega, Pi);
    P = F*P0*F.' + Q ;


    %% Update

    S = H*P*H.' + R + 1e-6*eye(3); % adding regularisation...
    K = P*H.' / S; 
    x(:,i+1) = x(:,i+1) + K*(w_measured - x(5:7,i+1).').';
    P = (eye(7) - K*H)*P;

    %% Normalise quaternions
    x(1:4, i+1) = x(1:4, i+1) / norm(x(1:4, i+1));

    P0 = P;
end


%% Graphing Results

figure(1)
plot(time, wx_true, "Color", "red")
hold on
plot(time, x(5,1:1000), "Color","blue")

figure(2)
plot(time, wy_true, "Color", "red")
hold on
plot(time, x(6,1:1000), "Color","blue")

figure(3)
plot(time, wz_true, "Color", "red")
hold on
plot(time, x(7,1:1000), "Color","blue")



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
