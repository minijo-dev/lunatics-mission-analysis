
% observation/measurement vector - accelerometer (3x1)
% b = [1;1;1];
b = [1;0;0];

% reference vector - magnetometer (3x1)
% r = [0.8;1;0];
r = [0.0001;1;0];

% weighting per observation vector
a = 1;

% attitude profile matrix
B = a*b*r.';

% components for K matrix
sigma = trace(B); % scalar
S = B + B.'; % (3x3)
Z = a*(cross(b,r)); % (3x1)

% K matrix
K = [S-sigma*eye(3) Z;
     Z.' sigma];

% characterisitic equation helpers
nu = 0.5*trace(S); % equal to sigma as long as B is square
k = trace(adjoint(S));
Delta = det(S);

e = nu^2 - k;
f = nu^2 + Z.'*Z;
g = Delta + Z.'*S*Z;
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
% q = [X; gamma];
q = [gamma; X];

% normalise quaternion
q = q / norm(q);

% Expected quaternion for 90-degree rotation around Z-axis
q_expected = [0; 0; 1/sqrt(2); 1/sqrt(2)];

% Compare the result with the expected quaternion
disp('Estimated Quaternion:');
disp(q);
disp('Expected Quaternion:');
disp(q_expected);

