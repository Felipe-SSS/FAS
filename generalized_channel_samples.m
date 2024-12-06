clc
close all
clear 

% SNR 
gamma = 1;
% SNR threshold
gamma_th = db2pow(5);
% Antenna length factor
W = 3;
% Sigma
sigma = sqrt(10);
% Number of UEs (antennas in the BS)
num_ues = 1;
% Number of samples
num_en = 10;
% Number of ports of FAS
N = 100; 
p_out = zeros(length(gamma_th), 1);

for g = 1:length( gamma_th )
    % Correlation matrix
    A = compute_corr_matrix(sigma, N, W);
    % Channels for each UE
    G_u = zeros(num_ues, num_en, N);
    % SINR for each UE
    SINR_u = zeros(num_ues, num_en, N);
    for u = 1:num_ues  
        G_u(u,:,:) = ch_gain_sp(N, num_en, A);
    end
    SINR_u( u,:, :) = abs(gamma^2 * G_u(u,:,:).^2);
    
    SINR = [];
    for u = 1 : num_ues
        local_sinr = reshape(SINR_u(u,:,:), num_en, N);
        SINR = [SINR; local_sinr];
        p_out(g) = p_out(g) + sum(max(local_sinr, [], 2) < gamma_th(g), 'all' ) / (num_en * num_ues);
    end
    writematrix(SINR, 'generalized_channel_samples_SINR.csv');
end

function A = compute_corr_matrix(sigma, num_ports, w)
    PHI = zeros(num_ports, num_ports);
    for k = 1 : num_ports
        for j = 1 : num_ports
            PHI(k, j) = sigma^2 * besselj(0, 2 * pi * (k - j) * w / (num_ports - 1));
        end
    end
    % Eigen decomposition
    [V, D] = eig(PHI);
    A = V * sqrtm(D);
end


function G = ch_gain_sp(num_ports, num_en, A)
    % Gaussian random
    X = normrnd(0, 1/sqrt(2), [num_en, num_ports] ) + 1i * normrnd(0, 1/sqrt(2), [num_en, num_ports]);
    % Gain matrix
    G = zeros(num_en, num_ports);
    for e = 1 : num_en
       G(e, :) =  A * X(e, :)';
    end
end