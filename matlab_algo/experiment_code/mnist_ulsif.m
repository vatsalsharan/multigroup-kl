% demo_KLIEP.m
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/KLIEP/

% clear all

rand('state',0);
randn('state',0);

%%%%%%%%%%%%%%%%%%%%%%%%% Generating data

% load('/Users/vatsalsharan/multicalibration/mnist_final_data/mnist9.mat')

[d,~] = size(data_p);
k = d;
subgroup = 1;

L = length(data_r_test);
if subgroup
    test_og = data_r_test;
    data_r_test = [data_r_test all_digits];
end

x_de = data_p;
x_nu = data_r;
x_disp = data_r_test;


x_de = normc(x_de);
x_nu = normc(x_nu);
x_disp = normc(x_disp);

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating density ratio
% [wh_x_de,wh_xdisp] = KLIEP(x_de,x_nu);
[wh_x_de,wh_xdisp] = uLSIF(x_de,x_nu, x_disp, 0.4, [], [], []);

log_w = log2(wh_xdisp);

ulsif_kl = mean(log_w(1:L))

if subgroup
    log_w = log_w(L+1:end);

    per_comp = 1200;
    ulsif_subgroup = zeros(10,1);
    for i = 1:10
        ulsif_subgroup(i) = mean(log_w(per_comp*(i-1)+1:per_comp*i));
    end
    data_r_test = test_og;
end
