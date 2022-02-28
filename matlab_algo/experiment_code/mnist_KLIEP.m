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
subgroup = 0;

L = length(data_r_test);
if subgroup
    test_og = data_r_test;
    data_r_test = [data_r_test all_digits];
end

x_de = data_p;
x_nu = data_r;
x_disp = data_r_test;
% 
% x_de2 = 0*x_de;
% x_nu2 = 0*x_nu;
% x_disp2 = 0*x_disp;
% 
% for i = 1:d
%     u1 = randi(d);
%     u2 = randi(d);
%     x_de2(i,:) = x_de(u1,:) .* x_de(u2,:);
%     x_nu2(i,:) = x_nu(u1,:) .* x_nu(u2,:);
%     x_disp2(i,:) = x_disp(u1,:) .* x_disp(u2,:);
% end
% 
% x_de = [x_de; x_de2];
% x_nu = [x_nu; x_nu2];
% x_disp = [x_disp; x_disp2];

x_de = normc(x_de);
x_nu = normc(x_nu);
x_disp = normc(x_disp);

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating density ratio
% [wh_x_de,wh_xdisp] = KLIEP(x_de,x_nu);
[wh_x_de,wh_xdisp] = KLIEP(x_de,x_nu, x_disp, [], [], []);

log_w = log2(wh_xdisp);

kliep_kl = mean(log_w(1:L))

if subgroup
    log_w = log_w(L+1:end);

    per_comp = 1200;
    kliep_subgroup = zeros(10,1);
    for i = 1:10
        kliep_subgroup(i) = mean(log_w(per_comp*(i-1)+1:per_comp*i));
    end
    data_r_test = test_og;
end
