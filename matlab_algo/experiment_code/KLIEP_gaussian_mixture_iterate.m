% rand('state',0);
% randn('state',0);

K = [5];
repeats = 2;

for k_counter=1:length(K)
    
    k = K(k_counter)
    kliep_kl = zeros(repeats,1);
    kliep_kl_std = zeros(repeats,1);
    
    for count = 1:repeats

        N = 30000;
        d = 2;
        delta = 2.5*ones(k,d);
        means_p = sqrt(k*10000)*randn(k,d);
        means_r = means_p + delta;
      
        x_de = return_p_sample(N, k, means_p, d);
        x_nu = return_r_sample(N, k, means_r, d);

        xdisp = return_r_sample(N, k, means_r, d);
        [wh_x_de,wh_xdisp]=KLIEP(x_de',x_nu',xdisp');

        log_w = log2(wh_xdisp);
        kliep_kl(count) = mean(log_w);

        kliep_subgroup = zeros(k,1);
        for i = 1:k
            per_comp = floor(N/k);
            kliep_subgroup(i) = mean(log_w(per_comp*(i-1)+1:per_comp*i));
        end

        kliep_kl_std(count) = std(kliep_subgroup);
    
    end
    
    str = strcat('kliep_30k_d',int2str(d),'_k',int2str(k),'.mat');
    save(str, 'kliep_kl', 'kliep_subgroup', 'd', 'k', 'delta', 'means_p', 'means_r', 'kliep_kl_std', 'N');

end

function [data_p] = return_p_sample(N, k, means_p, d)
    per_comp = floor(N/k);
    data_p = means_p(1,:) + randn(per_comp,d);
    for i = 2:k
        add_data = means_p(i,:) + randn(per_comp,d);
        data_p = [data_p; add_data];
    end
end    

function [data_r] = return_r_sample(N, k, means_r, d)
    per_comp = floor(N/k);
    data_r = means_r(1,:) + randn(per_comp,d);
    for i = 2:k
        add_data = means_r(i,:) + randn(per_comp,d);
        data_r = [data_r; add_data];
    end
end