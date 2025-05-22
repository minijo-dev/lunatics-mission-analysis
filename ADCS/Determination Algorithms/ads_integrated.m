%% EKF for Demonstration

% Setup Values - aeronautical building
lat = deg2rad(-33.88893628858434);
long = deg2rad(151.19446182380247);
alt = 19.8;

R_ECI = eci_vector(lat, long);
a_known = 9.81* R_ECI/norm(R_ECI); % known acceleration vector at aero building in 

% [m_known,~,~,~,~,~,~,~,~,~] = igrfmagm(alt, lat, long,decyear(2024,5,18));% known magnetometer vector in aero building
% m_known = m_known.'*; gives it in NED frame
m_known = [0;0;1];




% constants
Constant = ConstantClass(); 
Constant.a = [0.5, 0.5]; % weighting matrix
Constant.J = diag([0.0017, 0.0015, 0.0022]); % moment of inertia matrix
% Constant.Q = eye(7)*1e-7^2; % model noise matrix

Constant.sigma_tau = 1e-7; % external torque disturbances
Constant.sigma_q = 1e-8; % uncertainty in the linearisation of model
Constant.sigma_Acc = 0.01; % uncertainty in accelerometer measurements
Constant.sigma_Mag = 0.01; % uncertainty in magnetometer measurements
Constant.R = [Constant.sigma_Acc^2*eye(3)   eye(3);
              eye(3)                        Constant.sigma_Mag^2*eye(3)]; % measurement noise covariance matrix

% initialisation
count = 0;
iter = 10000;
tStart = tic;

init_state_true = [0;0;0.6;0.8;0.05;-0.1;0.15];
nts_state = [1;0;0;0;0.1;0.1;0.1];
nts_P = eye(7);

q0_plot = [1];
q0_true_plot = [0];
time = [0];
activate = true;

figure(1)
hold on
q0_func = plot(time,q0_plot,'-');
q0_func_true = plot(time,q0_true_plot,'-');

while count < 10000
    dt = 0.1;
    % dt = toc(tStart);
    % tStart = tic;

    %% True values for software testing only
    nts_state_true = StateTransitionFunction(init_state_true, init_state_true(5:7), dt, Constant);
    nts_state_true(1:4) = nts_state_true(1:4) / norm(nts_state_true(1:4));
    
    % Sensor input - SOFTWARE TEST ONLY
    C = inertial2body(nts_state_true);
    a_sensor = C*a_known + 0.001 * randn(3,1);
    m_sensor = C*m_known + 0.001 * randn(3,1);
    g_sensor = nts_state_true(5:7) + 0.001*randn(3,1) ;


    q0_true_plot(end+1) = nts_state_true(3);
    init_state_true = nts_state_true;
    

    %% Actual code
    % checking to see if QUEST quaternions should be used
    if count == 0 & activate == true
    % if mod(count,iter) == 0 & activate == true
        quest_quat = QUEST(a_known, m_known, a_sensor, m_sensor, Constant);
        init_state = [quest_quat(1);
                      quest_quat(2);
                      quest_quat(3); 
                      quest_quat(4); 
                      nts_state(5); 
                      nts_state(6);
                      nts_state(7)]; %(7x1)

        init_P = eye(7);
        disp("QUEST executed at " + string(datetime('now', 'Format', 'HH:mm:ss')));

    else
        init_state = nts_state;
        % init_state(5:7) = nts_state_true(5:7) + 0.01*randn(3,1);
        init_P = nts_P;
    end
    
    
    [nts_state, nts_P, tStart] = ekf(init_state, init_P, a_known, m_known,  a_sensor, m_sensor, g_sensor, dt, Constant);
    % disp("Estimated State: ")
    % disp(nts_state)

    count = count + 1;
 
    %% Plotting
    q0_plot(end+1) = nts_state(3);
    time(end+1) = time(end) + dt;

    % set(q0_func, 'XData', time, 'YData', q0_plot);
    % hold on
    % set(q0_func_true, 'XData', time, 'YData', q0_true_plot);
    % drawnow;
    % 

end

plot(time,q0_plot, 'Color','blue')
hold on
plot(time, q0_true_plot,'Color','red')

%% Functions

% QUEST
function quest_quat = QUEST(a_known, m_known, a_sensor, m_sensor, Constant)
    a = Constant.a;

    % attitude profile matrix
    B = a(1)*a_sensor*a_known.' + a(2)*m_sensor*m_known.';

    % components for K matrix
    sigma = trace(B);
    S = B + B.'; %(3x3)
    Z = a(1)*(cross(a_sensor,a_known)) + a(2)*(cross(m_sensor,m_known)); % (3x1)

    K = [S-sigma*eye(3) Z;
        Z.' sigma];

    % characterisitic equation helpers
    nu = 0.5*trace(S);
    k = trace(adjoint(S));
    Delta = det(S);

    e = nu^2 - k;
    f = nu^2 + Z.'*Z;
    g = Delta +Z.'*S*Z;
    h = Z.'*S^2*Z;

    % Newton Raphson Method
    lambda = sigma + norm(Z); % initial guess
    tol = 1e-7;
    error = 1;
    max_n = 0;

    while abs(error) > 1e-7
    % check if max iterations has been reached
        if max_n > 1000
            fprintf('Did not converge, error is %e\n', error)
            break
        end
    
        % characterisitic equation
        CE = lambda^4 - (e + f)*lambda^2 - g*lambda + (e*f +g*nu -h);
        % derivative of charcterisitic equation
        dCE = 4*lambda^3 - 2*(e+f)*lambda - g;
    
        lambda_np1 = lambda - CE/dCE;
        error = lambda_np1 - lambda;
    
        % update
        lambda = lambda_np1;
        max_n = max_n + 1;
    end

    % estimated quaternion helpers
    alpha = lambda^2 - sigma^2 + k;
    beta = lambda - sigma;
    
    X = (alpha*eye(3) + beta*S + S^2)*Z;
    gamma = (lambda+ sigma) * alpha - Delta;
    
    % estimated quaternion
    q = [gamma; X];
    
    % normalise quaternion
    quest_quat = q / norm(q);
    
 
end


% EKF
function [nts_state, nts_P, tStart] = ekf(init_state, init_P, a_known, m_known, a_sensor, m_sensor, g_sensor, dt, Constant)
    
    %% Predict
    % calculating next time step state vector
    nts_state = StateTransitionFunction(init_state, g_sensor, dt, Constant);

    % calculating next time step covariance matrix
    F = StateTransitionMatrix(init_state, Constant);
    Q = ProcessNoiseMatrix(dt, Constant);
    nts_P = F*init_P*F.' + Q;

    %% Update

    z = [a_sensor; 
         m_sensor]; % measurement vector in sensor body frame

    C = inertial2body(nts_state); % interial to body frame rotation matrix
    a_expected = C*a_known;
    m_expected = C*m_known; % assuming constant known values
    h = [a_expected;
         m_expected]; % measurement model in sensor body frame
    %% NEED TO CHANGE TO CUBESAT BODY FRAME - need sensor to cubesat body
    
    
    H = MeasurementModelJacobian(h, nts_state);

    v = z - h;
    S = H*nts_P*H.' + Constant.R;
    K = nts_P*H.'* inv(S); % Kalman gain

    % correction
    nts_state = nts_state + K*v;
    nts_P = (eye(7) - K*H)*nts_P; 

    % normalise quaternions
    nts_state(1:4) = nts_state(1:4) / norm(nts_state(1:4)); 

    tStart = tic;

end

%% EKF Helpers

% Captial Xi
function Xi = capital_xi(init_state)
    q0 = init_state(1);
    q1 = init_state(2);
    q2 = init_state(3);
    q3 = init_state(4);

    Xi = [-q1 -q2 -q3;
           q0 -q3 q2;
           q3 q0 -q1;
          -q2 q1 q0];
end

% State Transition Function
function nts_state = StateTransitionFunction(init_state, g_sensor, dt, Constant)
    
    J = Constant.J;
    omega = init_state(5:7); %(3x1)
    Omega = capital_omega(g_sensor); 
    % Xi = capital_xi(init_state(1:4));

    % qdot = 0.5*Xi*omega; %(4x1)
    qdot = 0.5*Omega*init_state(1:4);
    omega_dot = -J\(Constant.sigma_tau*ones(3,1) - cross(omega,(J*omega))); %(3x1)

    % state transition function
    f = [qdot; 
         omega_dot];

    nts_state = init_state + f*dt; %linearisation over small timestep

end


% Captial Omega
function Omega = capital_omega(omega)
    
    w = omega;
    Omega = [0 -w(1) -w(2) -w(3);
            w(1) 0 w(3) -w(2);
            w(2) -w(3) 0 w(1);
            w(3) w(2) -w(1) 0];

end

function Pi = capital_pi(init_state, Constant)
    J = Constant.J;
    J1 = J(1,1);
    J2 = J(2,2);
    J3 = J(3,3);

    J23 = J2 - J3;
    J31 = J3 - J1;
    J12 = J1 - J2;

    w = init_state(5:7);

    Pi = [0             (J23/J1)*w(3) (J23/J1)*w(2);
          (J31/J2)*w(3) 0             (J31/J2)*w(1);
          (J12/J3)*w(2) (J12/J3)*w(1) 0            ];

end

% State Transition Matrix
function F = StateTransitionMatrix(init_state, Constant)

    Omega = capital_omega(init_state);
    Xi = capital_xi(init_state);
    Pi = capital_pi(init_state, Constant);
    F = [0.5*Omega  0.5*Xi;
         zeros(3,4) Pi];

end


% Process Noise Covariance Matrix
function Q = ProcessNoiseMatrix(dt, Constant)

    Qq = Constant.sigma_q^2 * eye(4);
    Qw = Constant.sigma_tau^2 * Constant.J^-1 * (Constant.J^-1).' *dt;
    Q = [Qq          zeros(4,3);
         zeros(3,4)  Qw];
 
end

% Interial frame to body frame rotation matrix
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


% Jacobian for sensors
function H_sensor = SensorJacobian(sensor_vector, nts_state)

    x = sensor_vector(1);
    y = sensor_vector(2);
    z = sensor_vector(3);

    q0 = nts_state(1);
    q1 = nts_state(2);
    q2 = nts_state(3);
    q3 = nts_state(4);

    H_sensor = 2*[x*q0 + y*q3 - z*q2    x*q1 + y*q2 + z*q3    -x*q2 + y*q1 - z*q0    -x*q3 + y*q0 + z*q1;
                  -x*q3 + y*q0 + z*q1   x*q2 - y*q1 + z*q0    x*q1 + y*q2 + z*q3     -x*q0 - y*q3 + z*q2;
                  x*q2 - y*q1 + z*q0    x*q3 - y*q0 - z*q1    x*q0 + y*q3 - z*q2     x*q1 + y*q2 + z*q3];

end


% Measurement model Jacobian
function H = MeasurementModelJacobian(h, nts_state)

    H_acc = SensorJacobian(h(1:3),nts_state);
    H_mag = SensorJacobian(h(4:6),nts_state);

    H = [H_acc zeros(3,3);
         H_mag zeros(3,3)];
end

% Sensor body frame to CubeSat body frame
function S2C = sensor2sat(x,y,z)
% x,y,z are distances
    ...

end


% Surface ECI vector
function R_ECI = eci_vector(lat, long)

    Re = 6378*10e3;
    R_ECI = [Re*cos(lat)*cos(long);
             Re*cos(lat)*sin(long);
             Re*sin(lat)];

end
