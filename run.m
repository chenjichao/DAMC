function varargout = run(algo_name, dataset_name, varargin)
clc
close all;
warning('off','all');
diary off;
addpath('functions');
if exist(fullfile(pwd, 'spams', 'build'), 'dir')
    addpath(fullfile(pwd, 'spams', 'build'));
end
addpath('utils');
params = consts(algo_name, dataset_name, varargin{:});
if nargin == 0
    algo_name = 'DFAN';
    dataset_name = 'MSRCv1';
end
%% setup logging and snapshot path
time_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
if ~isempty(params.log_path)
    if ~exist(params.log_path, 'dir')
        mkdir(params.log_path);
    end
    diary(fullfile(params.log_path, sprintf('%s-%s-%s.log', algo_name, dataset_name, time_stamp)));
end
if ~isempty(params.cache_path) && ~exist(params.cache_path, 'dir')
    mkdir(params.cache_path);
end
%% parameters
disp('Parameters:');
disp(params);
fprintf('\n\n');
num_runs = params.num_runs;
iterations = params.iterations;
verbose = params.verbose;
parallel = params.parallel;
doplot = params.plot;
permdata = params.permdata;
cache_path = params.cache_path;
if params.seed > 0
    rng(params.seed);
else
    rng('shuffle');
end
params = rmfield(params, 'seed');
params = rmfield(params, 'plot');
params = rmfield(params, 'log_path');
params = rmfield(params, 'cache_path');
params = rmfield(params, 'permdata');


%% load data
addpath('data');
% addpath(genpath('data'));
load(dataset_name);
nCls = length(unique(Y));
nSmp = length(Y);
nViw = length(X);
% dataset statistics
fprintf('Dataset statistics of %s:\n', dataset_name);
fprintf('num_samples:  %d\n', nSmp);
fprintf('num_clusters: %d\n', nCls);
fprintf('num_features_each_view: \n');
for iViw = 1:nViw
    X{iViw} = normalize(X{iViw});
    fprintf('    %d\n', size(X{iViw}, 2));
end
fprintf('num_views: %d\n', nViw);

if permdata
    perm = randperm(nSmp);
    Y = Y(perm, :);
    for iViw = 1:nViw
        X{iViw} = normalize(X{iViw});
        X{iViw} = X{iViw}(perm, :); % size(X{iViw}) = [nSmp, nFtr];
    end
end

%% function handles
func_handle = eval(sprintf('@run%s', algo_name));
addpath('algos');
addpath('metrics');

%% get performance results
% nun_runs x param_sets x 3 (ACC/NMI/Purity)
tt = tic;
[results_mat, param_grid] = train_test(func_handle, X, Y, params);
fprintf('Done %s on %s in %.2fs.\n', algo_name, dataset_name, toc(tt));

fprintf('\n\n=================== Final Results of %s on %s =====================\n\n', algo_name, dataset_name);
acc_record = zeros(numel(param_grid), 7);
std_record = zeros(numel(param_grid), 7);
for i = 1:numel(param_grid)
    results_i = results_mat(:, i, :);
    mean_i = squeeze(mean(results_i, 1));
    std_i = squeeze(std(results_i, 0, 1));
    fprintf('Parameters %d/%d:\n', i, numel(param_grid));
    fprintf('ACC: %.4f (+-%.4f)\n', mean_i(1), std_i(1));
    fprintf('NMI: %.4f (+-%.4f)\n', mean_i(2), std_i(2));
    fprintf('PUR: %.4f (+-%.4f)\n', mean_i(3), std_i(3));
    fprintf('PRE: %.4f (+-%.4f)\n', mean_i(5), std_i(5));
    fprintf('REC: %.4f (+-%.4f)\n', mean_i(6), std_i(6));
    fprintf('F: %.4f (+-%.4f)\n', mean_i(4), std_i(4));
    fprintf('ARI: %.4f (+-%.4f)\n', mean_i(7), std_i(7));
    disp(param_grid(i));
    acc_record(i, :) = mean_i;
    std_record(i, :) = std_i;
end
[results_highest, selected] = max(acc_record, [], 1);
results_std = [std_record(selected(1), 1), std_record(selected(2), 2), std_record(selected(3), 3), std_record(selected(5), 5), std_record(selected(6), 6), std_record(selected(4), 4), std_record(selected(7), 7)];

fprintf('Highest accuracy for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(1), results_std(1));
best_params_acc = param_grid(selected(1));
disp(best_params_acc);

fprintf('Highest NMI for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(2), results_std(2));
best_params_nmi = param_grid(selected(2));
disp(best_params_nmi);

fprintf('Highest purity for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(3), results_std(3));
best_params_purity = param_grid(selected(3));
disp(best_params_purity);

fprintf('Highest precision for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(5), results_std(5));
best_params_precision = param_grid(selected(5));
disp(best_params_precision);

fprintf('Highest recall for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(6), results_std(6));
best_params_recall = param_grid(selected(6));
disp(best_params_recall);

fprintf('Highest fscore for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(4), results_std(4));
best_params_fscore = param_grid(selected(4));
disp(best_params_fscore);

fprintf('Highest ARI for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(7), results_std(7));
best_params_ARI = param_grid(selected(7));
disp(best_params_ARI);

results_highest = results_highest*100;
results_std = results_std *100;

% fprintf('for recording:\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n', ...
%     results_highest(1), results_std(1), ...
%     results_highest(2), results_std(2), ...
%     results_highest(3), results_std(3), ...
%     results_highest(5), results_std(5), ...
%     results_highest(6), results_std(6), ...
%     results_highest(4), results_std(4), ...
%     results_highest(7), results_std(7));

fprintf('for recording:\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n%.2f(+-%.2f)\n', ...
    results_highest(1), results_std(1), ...
    results_highest(2), results_std(2), ...
    results_highest(3), results_std(3), ...
    results_highest(5), results_std(5), ...
    results_highest(6), results_std(6), ...
    results_highest(4), results_std(4), ...
    results_highest(7), results_std(7));

if exist(cache_path, 'dir')
    save(fullfile(cache_path, sprintf('results-%s-%s-%s.mat', algo_name, dataset_name, time_stamp)), ...
        'results_mat', 'param_grid');
    save(fullfile(cache_path, sprintf('best_params-%s-%s-%s.mat', algo_name, dataset_name, time_stamp)), ...
        'best_params_acc', 'best_params_nmi', 'best_params_purity', 'results_highest', 'results_std');
end
fprintf('\n================= Final Results of %s on %s =================\n\n', algo_name, dataset_name);
%% optionally plot accuracy results
if length(doplot) == 2
    plot_acc_bar(param_grid, results_mat(:, :, 1), doplot{1}, doplot{2});
end
%% post-processing
diary off;
%% return values
if nargout > 0
    varargout{1} = results_highest;
end
if nargout > 1
    varargout{2} = [std_record(selected(1), 1), std_record(selected(2), 2), std_record(selected(3), 3)];
end
if nargout > 2
    varargout{3} = best_params_acc;
end
end




%% methods

% function labels = runMVGL(X, Y, params)
% nCls = length(unique(Y));
% nNbr = getfield_with_default(params, 'nNbr', 10);
% islocal_1 = getfield_with_default(params, 'islocal_1', 1);
% islocal_2 = getfield_with_default(params, 'islocal_2', 1);
% 
% labels = MVGL(X, nCls, nNbr, islocal_1, islocal_2);
% end

function labels = runCMSC(X, Y, params)
nCls = length(unique(Y));
lambda = getfield_with_default(params, 'lambda', 0.5); % 0.01-0.05
% In original experiments, the co-regularization parameter ? is varied from 0.01 to 0.05 

labels = CMSC(X, nCls, lambda);
end

function labels = runMMSC(X, Y, params)
nCls = length(unique(Y));
alpha = getfield_with_default(params, 'alpha', 0); % 1e[-2:0.2:2]
alpha = 10^alpha;
nNbr = getfield_with_default(params, 'nNbr', 5);

labels = MMSC(X, nCls, alpha, nNbr);
end

function labels = runMLAN(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);

labels = MLAN(X, nCls, Y, nNbr);
end

function labels = runMCGC(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);
beta = getfield_with_default(params, 'beta', 0); % 0.6
beta = 10^beta;

labels = MCGC(X, nCls, nNbr, beta);
end

function labels = runGSF(X, Y, params)
nCls = length(unique(Y));
% nNbr1 = getfield_with_default(params, 'nNbr1', 10);
nSmp = size(X{1}, 1);
% nNbr1 = nSmp - nNbr1;
snr = getfield_with_default(params, 'snr', 0.9); % 0.1:0.1:0.9
nNbr1 = floor(snr * nSmp);
nNbr2 = getfield_with_default(params, 'nNbr2', 10); % 3:10

var1 = getfield_with_default(params, 'var1', 1);
var2 = getfield_with_default(params, 'var2', 1); % auto-tuned in GSF()

for v = 1:length(X)
    S(:,:,v) = constructW_PKN(X{v}', nNbr1);
end
labels = GSF(S, nCls, var1, var2, nNbr2);
end

function labels = runDFAN(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);
fusion1 = getfield_with_default(params, 'fusion1', 'gm');
fusion2 = getfield_with_default(params, 'fusion2', 'am');

labels = DFAN(X, nCls, nNbr, fusion1, fusion2);
end

function labels = runELMAN(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);
fusion1 = getfield_with_default(params, 'fusion1', 'gm');
fusion2 = getfield_with_default(params, 'fusion2', 'am');
alpha = getfield_with_default(params, 'alpha', 0);
alpha = 10^(alpha);
nOut = getfield_with_default(params, 'nOut', 100);

labels = ELMAN(X, nCls, nNbr, fusion1, fusion2, alpha, nOut);
end

function labels = runELMDFAN_fixed(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);
fusion1 = getfield_with_default(params, 'fusion1', 'gm');
fusion2 = getfield_with_default(params, 'fusion2', 'am');
alpha = getfield_with_default(params, 'alpha', 0);
alpha = 10^(alpha);
nOut = getfield_with_default(params, 'nOut', 100);

labels = ELMDFAN_fixed(X, nCls, nNbr, fusion1, fusion2, alpha, nOut);
end

function labels = runELMDFAN_tuned(X, Y, params)
nCls = length(unique(Y));
nNbr = getfield_with_default(params, 'nNbr', 10);
fusion1 = getfield_with_default(params, 'fusion1', 'gm');
fusion2 = getfield_with_default(params, 'fusion2', 'am');
alpha = getfield_with_default(params, 'alpha', 0);
alpha = 10^(alpha);
nOut = getfield_with_default(params, 'nOut', 100);

labels = ELMDFAN_tuned(X, nCls, nNbr, fusion1, fusion2, alpha, nOut);
end
