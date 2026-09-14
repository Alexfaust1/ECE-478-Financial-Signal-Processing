

% Alexander Faust
% May 10, 2025
% Stochastic Calculus Problem Set
% 
% This script relied on the use of Claude 3.7 Sonnet for generation purposes. 
% I have modified sections of its output depending on the quality of output.

% Set random seed for reproducibility
rng(42);

% Sample size
N = 1e6;

% Generate samples
x_normal = randn(N, 1);                    % Standard normal N(0,1)
x_cauchy = 1 * tan(pi * (rand(N, 1) - 0.5)); % Cauchy with α = 1
x_t5 = trnd(5, N, 1) * sqrt(5/3);          % t with ν = 5, normalized to var = 1
x_t10 = trnd(10, N, 1) * sqrt(10/8);       % t with ν = 10, normalized to var = 1

% Calculate fraction of values where |x| > 4
frac_normal = mean(abs(x_normal) > 4);
frac_cauchy = mean(abs(x_cauchy) > 4);
frac_t5 = mean(abs(x_t5) > 4);
frac_t10 = mean(abs(x_t10) > 4);

% Display results
fprintf('Fraction |x| > 4 for Normal: %.6f\n', frac_normal);
fprintf('Fraction |x| > 4 for Cauchy: %.6f\n', frac_cauchy);
fprintf('Fraction |x| > 4 for t(5): %.6f\n', frac_t5);
fprintf('Fraction |x| > 4 for t(10): %.6f\n', frac_t10);

% Plot the first 1000 samples to visualize (easier to see than all 1e6)
figure;
subplot(4,1,1);
plot(1:1000, x_normal(1:1000), 'b.');
title('Normal Distribution N(0,1)');
ylim([-10 10]); % Initial view, may need adjustment

subplot(4,1,2);
plot(1:1000, x_cauchy(1:1000), 'r.');
title('Cauchy Distribution (α = 1)');
ylim([-50 50]); % Initial view, may need adjustment

subplot(4,1,3);
plot(1:1000, x_t5(1:1000), 'g.');
title('Student t Distribution (ν = 5, normalized)');
ylim([-10 10]); % Initial view, may need adjustment

subplot(4,1,4);
plot(1:1000, x_t10(1:1000), 'm.');
title('Student t Distribution (ν = 10, normalized)');
ylim([-10 10]); % Initial view, may need adjustment

% Create another plot with adjusted y-limits to see extreme values
figure;
subplot(4,1,1);
plot(1:1000, x_normal(1:1000), 'b.');
title('Normal Distribution N(0,1) - Adjusted Scale');
ylabel('Value');
ylim([-100 100]);

subplot(4,1,2);
plot(1:1000, x_cauchy(1:1000), 'r.');
title('Cauchy Distribution (α = 1) - Adjusted Scale');
ylabel('Value');
ylim([-100 100]);

subplot(4,1,3);
plot(1:1000, x_t5(1:1000), 'g.');
title('Student t Distribution (ν = 5, normalized) - Adjusted Scale');
ylabel('Value');
ylim([-100 100]);

subplot(4,1,4);
plot(1:1000, x_t10(1:1000), 'm.');
title('Student t Distribution (ν = 10, normalized) - Adjusted Scale');
xlabel('Sample Index');
ylabel('Value');
ylim([-100 100]);

% Comments;
% These plots illustrate how different distributions handle tail events.
% 
% Some distributions have far greater deviations such as cauchy.

% Part 2: AR and ARMA
% (a) Generate data using an ARMA(2,2) model
% ARMA(2,2) model: r_t = [(1 + 0.2z^-1)(1 + 0.5z^-1)]/[(1 - 0.8z^-1)(1 - 0.7z^-1)] * v_t

% Calculate MA coefficients by multiplying (1 + 0.2z^-1)(1 + 0.5z^-1)
ma = conv([1, 0.2], [1, 0.5]);  % = [1, 0.7, 0.1]

% Calculate AR coefficients by multiplying (1 - 0.8z^-1)(1 - 0.7z^-1)
ar = conv([1, -0.8], [1, -0.7]);  % = [1, -1.5, 0.56]

% Generate N = 250 samples
N = 250;
v_t = randn(N+100, 1);  % Generate noise samples with burn-in
r_t = filter(ma, ar, v_t);  % Apply ARMA filter
r_t = r_t(101:end);  % Remove burn-in period

% Part (a): Estimate covariances and autocorrelation coefficients
M = 10;  % Maximum order
[acf_values, lags] = autocorr(r_t, M);
gamma = acf_values * var(r_t);  % Convert to covariance by multiplying by variance
rho = acf_values;  % Autocorrelation coefficients

% Create stem plot
figure;
stem(lags, rho);
hold on;
yline(0.2, 'r--', 'Significance threshold');
yline(-0.2, 'r--');
xlabel('Lag m');
ylabel('Autocorrelation \rho(m)');
title('Stem Plot of Autocorrelation Coefficients');
grid on;

% Count significant lags
significant_lags = sum(abs(rho(2:end)) > 0.2);
fprintf('Number of significant lags (>0.2): %d\n', significant_lags);

% Part (b): Set up Toeplitz matrix and compute eigenvalues
C = toeplitz(gamma(1:M+1));
eigenvalues = eig(C);
fprintf('\nEigenvalues of Toeplitz matrix C:\n');
disp(eigenvalues);
fprintf('All eigenvalues positive: %d\n', all(eigenvalues > 0));

% Part (c): Find LDL decomposition of covariance matrix C
[L, D] = ldl(C);
fprintf('\nL from LDL decomposition:\n');
disp(round(L, 4));
fprintf('\nD from LDL decomposition:\n');
disp(round(D, 4));

% Part (d): Use Levinson-Durbin recursion
reflection_coeffs = zeros(M, 1);
prediction_errors = zeros(M, 1);
ar_coeffs = cell(M, 1);

% Initialize F matrix for FPEF coefficients
F = eye(M+1);

% Loop through AR orders 1 to M
for p = 1:M
    % Using Levinson-Durbin algorithm
    [a, e, k] = levinson(gamma(1:p+1), p);
    
    % Store the AR coefficients (excluding the leading 1)
    ar_coeffs{p} = a(2:end);
    
    reflection_coeffs(p) = k(end);
    prediction_errors(p) = e;
    
    % Fill in the F matrix (FPEF coefficients)
    % Use only the actual AR coefficients (excluding the leading 1)
    F(p+1, 1:p) = fliplr(a(2:end));
end

fprintf('\nLevinson-Durbin reflection coefficients:\n');
disp(round(reflection_coeffs, 4));
fprintf('\nPrediction error powers P_m:\n');
disp(round(prediction_errors, 4));

% Part (e): Compute FCF^T and compare
FCF_T = F * C * F';
fprintf('\nDiagonal elements of FCF^T:\n');
disp(round(diag(FCF_T), 4));
fprintf('\nPrediction error powers P_m for comparison:\n');
disp(round([gamma(1); prediction_errors], 4));  % Include P_0 = gamma(0)

% Compare F with L^-1
L_inv = inv(L);
fprintf('\nComparison of F and L^-1:\n');
fprintf('F:\n');
disp(round(F, 4));
fprintf('L^-1:\n');
disp(round(L_inv, 4));

% Part (f): Compute AR coefficients using least squares
X = zeros(N-M, M);
for i = 1:M
    X(:, i) = r_t(M-i+1:N-i);
end
y = r_t(M+1:N);

% LS fit
ar_ls = X \ y;
fprintf('\nAR coefficients using LS fit:\n');
disp(round(ar_ls', 4));  % Transpose to match format

fprintf('\nAR coefficients from Levinson-Durbin (order M):\n');
disp(round(ar_coeffs{M}, 4));

% Part (g): Analysis of reflection coefficients
fprintf('\nReflection coefficients analysis:\n');
for i = 1:M
    fprintf('Order %d: %.4f', i, reflection_coeffs(i));
    if abs(reflection_coeffs(i)) > 0.9
        fprintf(' - Close to 1: Suggests near non-stationarity or high persistence\n');
    elseif abs(reflection_coeffs(i)) < 0.1
        fprintf(' - Close to 0: Suggests this order adds little predictive value\n');
    else
        fprintf('\n');
    end
end

% Part (h): Model validation for different AR orders
ar_orders = [2, 5, 10];
figure;

for i = 1:length(ar_orders)
    order = ar_orders(i);
    ar_coef = ar_coeffs{order};
    
    % Create AR model and get residuals
    residuals = zeros(size(r_t));
    for t = order+1:length(r_t)
        prediction = ar_coef * r_t(t-1:-1:t-order);
        residuals(t) = r_t(t) - prediction;
    end
    
    % Compute autocorrelation of residuals
    [residual_acf, residual_lags] = autocorr(residuals(order+1:end), M);
    
    % Plot
    subplot(length(ar_orders), 1, i);
    stem(residual_lags, residual_acf);
    hold on;
    yline(0.2, 'r--');
    yline(-0.2, 'r--');
    title(sprintf('Residual Autocorrelation for AR(%d) Model', order));
    xlabel('Lag m');
    ylabel('\rho(m)');
    grid on;
    
    % Check if all |ρ(m)| < 0.2 for m ≠ 0
    is_white = all(abs(residual_acf(2:end)) < 0.2);
    fprintf('\nAR(%d) model - residuals are white noise: %d\n', order, is_white);
end

% Find optimal AR order by trial and error
if all(abs(residual_acf(2:end)) < 0.2)  % If AR(10) is already sufficient
    min_order = 10;
    for order = 2:9
        ar_coef = ar_coeffs{order};
        residuals = zeros(size(r_t));
        for t = order+1:length(r_t)
            prediction = ar_coef * r_t(t-1:-1:t-order);
            residuals(t) = r_t(t) - prediction;
        end
        [residual_acf, ~] = autocorr(residuals(order+1:end), M);
        if all(abs(residual_acf(2:end)) < 0.2)
            min_order = order;
            break;
        end
    end
    
    fprintf('\nOptimal AR order (smallest that passes whiteness test): %d\n', min_order);
end



% =======================================================================
%   ARIMA(2,1,2)
% ========================================================================

% Define ARMA(2,2) model parameters from Problem 2
ma_orig = conv([1, 0.2], [1, 0.5]);    % = [1, 0.7, 0.1]
ar_orig = conv([1, -0.8], [1, -0.7]);  % = [1, -1.5, 0.56]

% Create ARIMA(2,1,2) by adding near unit root (1 - 0.99z^-1)^-1
% This means we concatenate [1, -0.99] to the AR part
ar_arima = conv(ar_orig, [1, -0.99]);  % Adds near unit-root

% Generate samples
N = 250;
v_t = randn(N+100, 1);  % Generate noise with burn-in

% Generate original ARMA(2,2) series
r_t_orig = filter(ma_orig, ar_orig, v_t);
r_t_orig = r_t_orig(101:end);  % Remove burn-in

% Generate ARIMA(2,1,2) series with near unit-root
r_t_arima = filter(ma_orig, ar_arima, v_t);
r_t_arima = r_t_arima(101:end);  % Remove burn-in

% Plot both time series
figure;
subplot(2,1,1);
plot(r_t_orig);
title('Original ARMA(2,2) Series');
xlabel('Time');
ylabel('Value');
grid on;

subplot(2,1,2);
plot(r_t_arima);
title('ARIMA(2,1,2) with Near Unit-Root');
xlabel('Time');
ylabel('Value');
grid on;

% Compute first difference of ARIMA series
s_t = diff(r_t_arima);

% Plot the differenced series
figure;
plot(s_t);
title('First Difference of ARIMA Series (s_t = r_t - r_{t-1})');
xlabel('Time');
ylabel('Value');
grid on;

% Define maximum order
M = 10;

% Compute autocorrelation for original ARMA
[acf_orig, lags_orig] = autocorr(r_t_orig, M);
gamma_orig = acf_orig * var(r_t_orig);
rho_orig = acf_orig;

% Compute autocorrelation for ARIMA
[acf_arima, lags_arima] = autocorr(r_t_arima, M);
gamma_arima = acf_arima * var(r_t_arima);
rho_arima = acf_arima;

% Compute autocorrelation for differenced series
[acf_diff, lags_diff] = autocorr(s_t, M);
gamma_diff = acf_diff * var(s_t);
rho_diff = acf_diff;

% Plot autocorrelation stems for comparison
figure;
subplot(3,1,1);
stem(lags_orig, rho_orig);
hold on;
yline(0.2, 'r--');
yline(-0.2, 'r--');
title('Autocorrelation of Original ARMA(2,2)');
xlabel('Lag m');
ylabel('\rho(m)');
grid on;

subplot(3,1,2);
stem(lags_arima, rho_arima);
hold on;
yline(0.2, 'r--');
yline(-0.2, 'r--');
title('Autocorrelation of ARIMA with Near Unit-Root');
xlabel('Lag m');
ylabel('\rho(m)');
grid on;

subplot(3,1,3);
stem(lags_diff, rho_diff);
hold on;
yline(0.2, 'r--');
yline(-0.2, 'r--');
title('Autocorrelation of Differenced Series s_t');
xlabel('Lag m');
ylabel('\rho(m)');
grid on;

% AR coefficient estimation for all three series
% Use only Levinson-Durbin for brevity
ar_orders = [2, 5, 10];

fprintf('======= AR Model Analysis =======\n');

% 1. Original ARMA Series
fprintf('\n----- Original ARMA(2,2) Series -----\n');
for order = ar_orders
    [a, e, k] = levinson(gamma_orig(1:order+1), order);
    fprintf('AR(%d) model: last reflection coefficient = %.4f, prediction error = %.4f\n', ...
            order, k(end), e);
end

% 2. ARIMA Series with near unit-root
fprintf('\n----- ARIMA(2,1,2) Series with Near Unit-Root -----\n');
try
    for order = ar_orders
        [a, e, k] = levinson(gamma_arima(1:order+1), order);
        fprintf('AR(%d) model: last reflection coefficient = %.4f, prediction error = %.4f\n', ...
                order, k(end), e);
    end
catch ME
    fprintf('Error in Levinson-Durbin for ARIMA series: %s\n', ME.message);
    fprintf('This may indicate issues with positive-definiteness due to nonstationarity.\n');
end

% 3. Differenced Series
fprintf('\n----- Differenced Series s_t -----\n');
for order = ar_orders
    [a, e, k] = levinson(gamma_diff(1:order+1), order);
    fprintf('AR(%d) model: last reflection coefficient = %.4f, prediction error = %.4f\n', ...
            order, k(end), e);
end

% Residual whiteness test for differenced series
fprintf('\n----- Residual Whiteness Test for Differenced Series -----\n');
for i = 1:length(ar_orders)
    order = ar_orders(i);
    [a, ~, ~] = levinson(gamma_diff(1:order+1), order);
    
    % Get AR coefficients (exclude the leading 1)
    ar_coef = a(2:end);
    
    % Create AR model and get residuals
    residuals = zeros(size(s_t));
    for t = order+1:length(s_t)
        prediction = ar_coef * s_t(t-1:-1:t-order);
        residuals(t) = s_t(t) - prediction;
    end
    
    % Test residual whiteness
    [residual_acf, ~] = autocorr(residuals(order+1:end), M);
    is_white = all(abs(residual_acf(2:end)) < 0.2);
    fprintf('AR(%d) model for differenced series - residuals are white noise: %d\n', ...
            order, is_white);
end

fprintf('\n======= Unit Root Detection Analysis =======\n');
fprintf('Correlation at lag 1 for ARIMA series: %.4f\n', rho_arima(2));
fprintf('Correlation at lag 1 for differenced series: %.4f\n', rho_diff(2));

% Compute the optimal AR order for the differenced series
if all(abs(residual_acf(2:end)) < 0.2)  % If AR(10) is sufficient
    min_order = 10;
    for order = 2:9
        [a, ~, ~] = levinson(gamma_diff(1:order+1), order);
        ar_coef = a(2:end);
        
        residuals = zeros(size(s_t));
        for t = order+1:length(s_t)
            prediction = ar_coef * s_t(t-1:-1:t-order);
            residuals(t) = s_t(t) - prediction;
        end
        
        [residual_acf, ~] = autocorr(residuals(order+1:end), M);
        if all(abs(residual_acf(2:end)) < 0.2)
            min_order = order;
            break;
        end
    end
    
    fprintf('Optimal AR order for differenced series: %d\n', min_order);
end



% =======================================================================
% GARCH/ARCH
% =======================================================================
%% Part 5(a): GARCH Simulation
% This part was fully generated by Claude... I will investigate this more
% over the summer!

% GARCH(1,1) parameters as specified with the correction noted (beta=0.3 instead of 0.4)
omega = 0.5;
alpha = 0.6;
beta1 = 0.3;  % Using the recommended value instead of 0.4
beta2 = 0.399;  % Will also try this value to see "if something weird happens"

% Number of observations
T = 1000;

% Initialize arrays for returns and volatility
r_norm_stable = zeros(T, 1);
sigma_norm_stable = zeros(T, 1);
r_t_stable = zeros(T, 1);
sigma_t_stable = zeros(T, 1);

r_norm_unstable = zeros(T, 1);
sigma_norm_unstable = zeros(T, 1);
r_t_unstable = zeros(T, 1);
sigma_t_unstable = zeros(T, 1);

% Initial values
sigma_norm_stable(1) = sqrt(omega / (1 - alpha - beta1));  % Unconditional variance
sigma_t_stable(1) = sigma_norm_stable(1);

sigma_norm_unstable(1) = sqrt(omega / (1 - alpha - beta2));  % This might not be valid
sigma_t_unstable(1) = sigma_norm_unstable(1);

% Generate standard normal and t(5) random variables
z_normal = randn(T, 1);
z_t = trnd(5, T, 1) / sqrt(5/3);  % Normalize to have variance = 1

% Simulate GARCH(1,1) processes with normal innovations - Stable version (beta=0.3)
for t = 2:T
    % Update volatility
    sigma_norm_stable(t) = sqrt(omega + alpha * r_norm_stable(t-1)^2 + beta1 * sigma_norm_stable(t-1)^2);
    
    % Generate return
    r_norm_stable(t) = sigma_norm_stable(t) * z_normal(t);
end

% Simulate GARCH(1,1) processes with t innovations - Stable version (beta=0.3)
for t = 2:T
    % Update volatility
    sigma_t_stable(t) = sqrt(omega + alpha * r_t_stable(t-1)^2 + beta1 * sigma_t_stable(t-1)^2);
    
    % Generate return
    r_t_stable(t) = sigma_t_stable(t) * z_t(t);
end

% Simulate GARCH(1,1) processes with normal innovations - Potentially unstable (beta=0.4)
for t = 2:T
    % Update volatility
    sigma_norm_unstable(t) = sqrt(omega + alpha * r_norm_unstable(t-1)^2 + beta2 * sigma_norm_unstable(t-1)^2);
    
    % Generate return
    r_norm_unstable(t) = sigma_norm_unstable(t) * z_normal(t);
end

% Simulate GARCH(1,1) processes with t innovations - Potentially unstable (beta=0.4)
for t = 2:T
    % Update volatility
    sigma_t_unstable(t) = sqrt(omega + alpha * r_t_unstable(t-1)^2 + beta2 * sigma_t_unstable(t-1)^2);
    
    % Generate return
    r_t_unstable(t) = sigma_t_unstable(t) * z_t(t);
end

% Burn-in period: Remove first 200 observations
burnin = 200;
r_norm_stable = r_norm_stable(burnin+1:end);
sigma_norm_stable = sigma_norm_stable(burnin+1:end);
r_t_stable = r_t_stable(burnin+1:end);
sigma_t_stable = sigma_t_stable(burnin+1:end);

r_norm_unstable = r_norm_unstable(burnin+1:end);
sigma_norm_unstable = sigma_norm_unstable(burnin+1:end);
r_t_unstable = r_t_unstable(burnin+1:end);
sigma_t_unstable = sigma_t_unstable(burnin+1:end);

% Plot returns and volatility - Stable case (beta=0.3)
figure;
subplot(2,1,1);
plot(r_norm_stable, 'b');
hold on;
plot(2*sigma_norm_stable, 'r', 'LineWidth', 1.5);
plot(-2*sigma_norm_stable, 'r', 'LineWidth', 1.5);
title('GARCH(1,1) with Normal Innovations (beta=0.3)');
xlabel('Time');
ylabel('Returns and Volatility');
legend('Returns r_t', '2\sigma_t', '-2\sigma_t', 'Location', 'best');

subplot(2,1,2);
plot(r_t_stable, 'b');
hold on;
plot(2*sigma_t_stable, 'r', 'LineWidth', 1.5);
plot(-2*sigma_t_stable, 'r', 'LineWidth', 1.5);
title('GARCH(1,1) with t(5) Innovations (beta=0.3)');
xlabel('Time');
ylabel('Returns and Volatility');
legend('Returns r_t', '2\sigma_t', '-2\sigma_t', 'Location', 'best');

% Plot returns and volatility - Potentially unstable case (beta=0.4)
figure;
subplot(2,1,1);
plot(r_norm_unstable, 'b');
hold on;
plot(2*sigma_norm_unstable, 'r', 'LineWidth', 1.5);
plot(-2*sigma_norm_unstable, 'r', 'LineWidth', 1.5);
title('GARCH(1,1) with Normal Innovations (beta=0.4, boundary case)');
xlabel('Time');
ylabel('Returns and Volatility');
legend('Returns r_t', '2\sigma_t', '-2\sigma_t', 'Location', 'best');

subplot(2,1,2);
plot(r_t_unstable, 'b');
hold on;
plot(2*sigma_t_unstable, 'r', 'LineWidth', 1.5);
plot(-2*sigma_t_unstable, 'r', 'LineWidth', 1.5);
title('GARCH(1,1) with t(5) Innovations (beta=0.4, boundary case)');
xlabel('Time');
ylabel('Returns and Volatility');
legend('Returns r_t', '2\sigma_t', '-2\sigma_t', 'Location', 'best');

% Compute and plot autocorrelation of returns and squared returns
figure;
subplot(2,2,1);
autocorr(r_norm_stable, 20);
title('ACF of Returns (Normal, beta=0.3)');

subplot(2,2,2);
autocorr(r_norm_stable.^2, 20);
title('ACF of Squared Returns (Normal, beta=0.3)');

subplot(2,2,3);
autocorr(r_t_stable, 20);
title('ACF of Returns (t(5), beta=0.3)');

subplot(2,2,4);
autocorr(r_t_stable.^2, 20);
title('ACF of Squared Returns (t(5), beta=0.3)');

% Do the same for the boundary case (beta=0.4)
figure;
subplot(2,2,1);
autocorr(r_norm_unstable, 20);
title('ACF of Returns (Normal, beta=0.4)');

subplot(2,2,2);
autocorr(r_norm_unstable.^2, 20);
title('ACF of Squared Returns (Normal, beta=0.4)');

subplot(2,2,3);
autocorr(r_t_unstable, 20);
title('ACF of Returns (t(5), beta=0.4)');

subplot(2,2,4);
autocorr(r_t_unstable.^2, 20);
title('ACF of Squared Returns (t(5), beta=0.4)');

%% Fit GARCH to simulated data
fprintf('Fitting GARCH(1,1) models to simulated data...\n');
    
% Fit GARCH(1,1) to normal innovations data
Mdl_norm = garch(1,1);
[EstMdl_norm, EstParamCov_norm] = estimate(Mdl_norm, r_norm_stable);
disp('GARCH(1,1) fit to Normal innovations data:');
disp(EstMdl_norm);

% Fit GARCH(1,1) to t innovations data
Mdl_t = garch(1,1);
[EstMdl_t, EstParamCov_t] = estimate(Mdl_t, r_t_stable);
disp('GARCH(1,1) fit to t(5) innovations data:');
disp(EstMdl_t);

% Fit ARCH(2) to normal innovations data
Mdl_arch_norm = garch(0,2);
[EstMdl_arch_norm, EstParamCov_arch_norm] = estimate(Mdl_arch_norm, r_norm_stable);
disp('ARCH(2) fit to Normal innovations data:');
disp(EstMdl_arch_norm);

% Fit ARCH(2) to t innovations data
Mdl_arch_t = garch(0,2);
[EstMdl_arch_t, EstParamCov_arch_t] = estimate(Mdl_arch_t, r_t_stable);
disp('ARCH(2) fit to t(5) innovations data:');
disp(EstMdl_arch_t);

try
    % Try the newer version syntax first
    [~, ~, Sigma_norm] = simulate(EstMdl_norm, length(r_norm_stable), 'E0', r_norm_stable(1));
    [~, ~, Sigma_t] = simulate(EstMdl_t, length(r_t_stable), 'E0', r_t_stable(1));
    [~, ~, Sigma_arch_norm] = simulate(EstMdl_arch_norm, length(r_norm_stable), 'E0', r_norm_stable(1));
    [~, ~, Sigma_arch_t] = simulate(EstMdl_arch_t, length(r_t_stable), 'E0', r_t_stable(1));
catch
    % Alternative approach for older MATLAB versions
    % Generate conditional variances directly from the model equations
    fprintf('Using alternative method to generate conditional variances...\n');
    
    % For GARCH(1,1) with normal innovations
    Sigma_norm = zeros(size(r_norm_stable));
    Sigma_norm(1) = var(r_norm_stable);  % Initialize with sample variance
    for t = 2:length(r_norm_stable)
        Sigma_norm(t) = EstMdl_norm.Constant + ...
                       sum(EstMdl_norm.ARCH .* r_norm_stable(t-1:-1:t-length(EstMdl_norm.ARCH)).^2) + ...
                       sum(EstMdl_norm.GARCH .* Sigma_norm(t-1:-1:t-length(EstMdl_norm.GARCH)));
    end
    
    % For GARCH(1,1) with t innovations
    Sigma_t = zeros(size(r_t_stable));
    Sigma_t(1) = var(r_t_stable);
    for t = 2:length(r_t_stable)
        Sigma_t(t) = EstMdl_t.Constant + ...
                    sum(EstMdl_t.ARCH .* r_t_stable(t-1:-1:t-length(EstMdl_t.ARCH)).^2) + ...
                    sum(EstMdl_t.GARCH .* Sigma_t(t-1:-1:t-length(EstMdl_t.GARCH)));
    end
    
    % For ARCH(2) with normal innovations
    Sigma_arch_norm = zeros(size(r_norm_stable));
    Sigma_arch_norm(1) = var(r_norm_stable);
    for t = 2:length(r_norm_stable)
        if t == 2
            % Special case for t=2 since we need two lagged returns
            Sigma_arch_norm(t) = EstMdl_arch_norm.Constant + ...
                               EstMdl_arch_norm.ARCH(1) * r_norm_stable(t-1)^2;
        else
            Sigma_arch_norm(t) = EstMdl_arch_norm.Constant + ...
                               sum(EstMdl_arch_norm.ARCH .* r_norm_stable(t-1:-1:t-length(EstMdl_arch_norm.ARCH)).^2);
        end
    end
    
    % For ARCH(2) with t innovations
    Sigma_arch_t = zeros(size(r_t_stable));
    Sigma_arch_t(1) = var(r_t_stable);
    for t = 2:length(r_t_stable)
        if t == 2
            Sigma_arch_t(t) = EstMdl_arch_t.Constant + ...
                            EstMdl_arch_t.ARCH(1) * r_t_stable(t-1)^2;
        else
            Sigma_arch_t(t) = EstMdl_arch_t.Constant + ...
                            sum(EstMdl_arch_t.ARCH .* r_t_stable(t-1:-1:t-length(EstMdl_arch_t.ARCH)).^2);
        end
    end
end
% Plot actual vs. estimated volatility
figure;
subplot(2,2,1);
plot(sigma_norm_stable, 'b', 'LineWidth', 1.5);
hold on;
plot(Sigma_norm, 'r--', 'LineWidth', 1);
title('Actual vs. Estimated Volatility (GARCH(1,1), Normal)');
legend('True \sigma_t', 'Estimated \sigma_t');

subplot(2,2,2);
plot(sigma_t_stable, 'b', 'LineWidth', 1.5);
hold on;
plot(Sigma_t, 'r--', 'LineWidth', 1);
title('Actual vs. Estimated Volatility (GARCH(1,1), t(5))');
legend('True \sigma_t', 'Estimated \sigma_t');

subplot(2,2,3);
plot(sigma_norm_stable, 'b', 'LineWidth', 1.5);
hold on;
plot(Sigma_arch_norm, 'r--', 'LineWidth', 1);
title('Actual vs. Estimated Volatility (ARCH(2), Normal)');
legend('True \sigma_t', 'Estimated \sigma_t');

subplot(2,2,4);
plot(sigma_t_stable, 'b', 'LineWidth', 1.5);
hold on;
plot(Sigma_arch_t, 'r--', 'LineWidth', 1);
title('Actual vs. Estimated Volatility (ARCH(2), t(5))');
legend('True \sigma_t', 'Estimated \sigma_t');


%% Part 5(b): Real Financial Data Analysis
try
    % Try to download historical data if internet is available
    fprintf('\nDownloading S&P 500 and stock data...\n');
    
    % Download S&P 500 data for 2 years
    sp500 = webread('https://query1.finance.yahoo.com/v7/finance/download/^GSPC?interval=1d&period1=1577836800&period2=1640995200&events=history');
    
    % Download Apple and Microsoft data for 2 years
    aapl = webread('https://query1.finance.yahoo.com/v7/finance/download/AAPL?interval=1d&period1=1577836800&period2=1640995200&events=history');
    msft = webread('https://query1.finance.yahoo.com/v7/finance/download/MSFT?interval=1d&period1=1577836800&period2=1640995200&events=history');
    
    % Compute log returns
    sp500_returns = diff(log(sp500.AdjClose));
    aapl_returns = diff(log(aapl.AdjClose));
    msft_returns = diff(log(msft.AdjClose));
    
    % Subtract mean
    sp500_returns = sp500_returns - mean(sp500_returns);
    aapl_returns = aapl_returns - mean(aapl_returns);
    msft_returns = msft_returns - mean(msft_returns);
    
    % Plot returns and squared returns
    figure;
    subplot(3,2,1);
    plot(sp500_returns);
    title('S&P 500 Log Returns');
    
    subplot(3,2,2);
    plot(sp500_returns.^2);
    title('S&P 500 Squared Log Returns');
    
    subplot(3,2,3);
    plot(aapl_returns);
    title('Apple Log Returns');
    
    subplot(3,2,4);
    plot(aapl_returns.^2);
    title('Apple Squared Log Returns');
    
    subplot(3,2,5);
    plot(msft_returns);
    title('Microsoft Log Returns');
    
    subplot(3,2,6);
    plot(msft_returns.^2);
    title('Microsoft Squared Log Returns');
    
    % Compute and plot autocorrelation of returns and squared returns
    figure;
    subplot(3,2,1);
    autocorr(sp500_returns, 20);
    title('ACF of S&P 500 Returns');
    
    subplot(3,2,2);
    autocorr(sp500_returns.^2, 20);
    title('ACF of S&P 500 Squared Returns');
    
    subplot(3,2,3);
    autocorr(aapl_returns, 20);
    title('ACF of Apple Returns');
    
    subplot(3,2,4);
    autocorr(aapl_returns.^2, 20);
    title('ACF of Apple Squared Returns');
    
    subplot(3,2,5);
    autocorr(msft_returns, 20);
    title('ACF of Microsoft Returns');
    
    subplot(3,2,6);
    autocorr(msft_returns.^2, 20);
    title('ACF of Microsoft Squared Returns');
    
    % Fit GARCH models to real data
    try
        % Fit GARCH(1,1) to S&P 500 returns
        fprintf('\nFitting GARCH models to real financial data...\n');
        
        Mdl_sp500_garch = garch(1,1);
        [EstMdl_sp500_garch, EstParamCov_sp500_garch] = estimate(Mdl_sp500_garch, sp500_returns);
        disp('GARCH(1,1) fit to S&P 500:');
        disp(EstMdl_sp500_garch);
        
        % Fit ARCH(2) to S&P 500 returns
        Mdl_sp500_arch = garch(0,2);
        [EstMdl_sp500_arch, EstParamCov_sp500_arch] = estimate(Mdl_sp500_arch, sp500_returns);
        disp('ARCH(2) fit to S&P 500:');
        disp(EstMdl_sp500_arch);
        
        % Generate conditional variances
        [~, ~, Sigma_sp500_garch] = simulate(EstMdl_sp500_garch, length(sp500_returns), 'E0', sp500_returns(1));
        [~, ~, Sigma_sp500_arch] = simulate(EstMdl_sp500_arch, length(sp500_returns), 'E0', sp500_returns(1));
        
        % Plot returns with estimated volatility
        figure;
        subplot(2,1,1);
        plot(sp500_returns, 'b');
        hold on;
        plot(2*sqrt(Sigma_sp500_garch), 'r', 'LineWidth', 1.5);
        plot(-2*sqrt(Sigma_sp500_garch), 'r', 'LineWidth', 1.5);
        title('S&P 500 Returns with GARCH(1,1) Volatility Envelope');
        xlabel('Time');
        ylabel('Returns');
        legend('Returns', '2\sigma_t', '-2\sigma_t');
        
        subplot(2,1,2);
        plot(sp500_returns, 'b');
        hold on;
        plot(2*sqrt(Sigma_sp500_arch), 'r', 'LineWidth', 1.5);
        plot(-2*sqrt(Sigma_sp500_arch), 'r', 'LineWidth', 1.5);
        title('S&P 500 Returns with ARCH(2) Volatility Envelope');
        xlabel('Time');
        ylabel('Returns');
        legend('Returns', '2\sigma_t', '-2\sigma_t');
        
        % Do the same for Apple and Microsoft
        % For brevity, we'll just do GARCH(1,1) for these
        
        % Apple
        Mdl_aapl_garch = garch(1,1);
        [EstMdl_aapl_garch, ~] = estimate(Mdl_aapl_garch, aapl_returns);
        [~, ~, Sigma_aapl_garch] = simulate(EstMdl_aapl_garch, length(aapl_returns), 'E0', aapl_returns(1));
        
        figure;
        plot(aapl_returns, 'b');
        hold on;
        plot(2*sqrt(Sigma_aapl_garch), 'r', 'LineWidth', 1.5);
        plot(-2*sqrt(Sigma_aapl_garch), 'r', 'LineWidth', 1.5);
        title('Apple Returns with GARCH(1,1) Volatility Envelope');
        xlabel('Time');
        ylabel('Returns');
        legend('Returns', '2\sigma_t', '-2\sigma_t');
        
        % Microsoft
        Mdl_msft_garch = garch(1,1);
        [EstMdl_msft_garch, ~] = estimate(Mdl_msft_garch, msft_returns);
        [~, ~, Sigma_msft_garch] = simulate(EstMdl_msft_garch, length(msft_returns), 'E0', msft_returns(1));
        
        figure;
        plot(msft_returns, 'b');
        hold on;
        plot(2*sqrt(Sigma_msft_garch), 'r', 'LineWidth', 1.5);
        plot(-2*sqrt(Sigma_msft_garch), 'r', 'LineWidth', 1.5);
        title('Microsoft Returns with GARCH(1,1) Volatility Envelope');
        xlabel('Time');
        ylabel('Returns');
        legend('Returns', '2\sigma_t', '-2\sigma_t');
        
    catch ME
        fprintf('Error fitting GARCH models to real data: %s\n', ME.message);
    end
    
catch ME
    fprintf('Error downloading financial data: %s\n', ME.message);
    fprintf('If you have internet connectivity issues, please download the data manually from Yahoo Finance.\n');
end

