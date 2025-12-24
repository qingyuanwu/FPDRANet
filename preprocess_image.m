function intensityMAP_image = preprocess_image(GT)
% PREPROCESS_IMAGE Preprocesses the input image using the SPIRAL-TAP algorithm.
%
%   intensityMAP_image = preprocess_image(GT) takes a Ground Truth (GT) or
%   input image, applies an inverse transformation, sets up wavelet basis
%   functions, and uses the SPIRAL-TAP modified algorithm to reconstruct
%   probability maps. The result is then normalized.
%
%   Inputs:
%       GT - Input image matrix (Ground Truth or Observation).
%
%   Outputs:
%       intensityMAP_image - The processed and normalized intensity map.
%
%   Dependencies:
%       - Rice Wavelet Toolbox (daubcqf, midwt, mdwt)
%       - SPIRAL Toolbox (SPIRALTAP_modified)

    % Add current directory and all subdirectories to the MATLAB path
    % to ensure dependencies (spiral, rwt, etc.) are found.
    addpath(genpath(pwd));

    % ---------------------------------------------------------------------
    % 1. Initialization and Inverse Transformation
    % ---------------------------------------------------------------------
    % Define scaling constant alpha
    alph = 1 / (7e-6);

    % Compute inverse of the input image
    prob_image = 1 ./ GT;

    % Compute initial Maximum Likelihood (ML) intensity estimate
    % intensityML = -alpha * log(1 - prob)
    intensityML_image = -alph * log(1 - prob_image);

    % ---------------------------------------------------------------------
    % 2. Wavelet Basis Configuration
    % ---------------------------------------------------------------------
    % Generate Daubechies filter coefficients (length 2)
    wav = daubcqf(2);

    % Define Inverse Discrete Wavelet Transform (IDWT) handle
    W = @(x) midwt(x, wav);

    % Define Discrete Wavelet Transform (DWT) handle
    WT = @(x) mdwt(x, wav);

    % ---------------------------------------------------------------------
    % 3. Algorithm Parameters
    % ---------------------------------------------------------------------
    tau = 0.3;          % Penalty parameter for regularization
    ainit = 0.09;       % Initial descent step size
    maxiter = 25;       % Maximum number of iterations
    set_penalty = 'tv'; % Type of penalty (Total Variation)

    % Define system matrices (Identity matrices in this case)
    AT = @(x) x;
    A  = @(x) x;

    % No censoring map used
    nMap = [];

    % ---------------------------------------------------------------------
    % 4. Run SPIRAL-TAP Algorithm
    % ---------------------------------------------------------------------
    % Execute the reconstruction algorithm
    prob_image = SPIRALTAP_modified(nMap, GT, A, tau, ...
        'noisetype', 'geometric', ...
        'penalty', set_penalty, ...
        'maxiter', maxiter, ...
        'Initialization', intensityML_image, ...
        'AT', AT, ...
        'monotone', 100, ...
        'miniter', 1, ...
        'W', W, ...
        'WT', WT, ...
        'stopcriterion', 3, ...
        'tolerance', 1e-1, ...
        'alphainit', ainit, ...
        'alphaaccept', 1e80, ...
        'logepsilon', 1e-10, ...
        'saveobjective', 1, ...
        'savereconerror', 1, ...
        'savecputime', 1, ...
        'savesolutionpath', 0, ...
        'truth', zeros(64, 64), ... % Placeholder for ground truth
        'verbose', 5);

    % ---------------------------------------------------------------------
    % 5. Post-processing and Normalization
    % ---------------------------------------------------------------------
    % Convert probability map back to intensity map
    intensityMAP_image = -alph * log(1 - exp(-prob_image));

    % Normalize the output image to the range [0, 1]
    % Shift minimum to 0
    intensityMAP_image = intensityMAP_image - min(intensityMAP_image(:));
    % Scale maximum to 1
    intensityMAP_image = intensityMAP_image / max(intensityMAP_image(:));

end