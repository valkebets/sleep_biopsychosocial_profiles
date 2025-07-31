%% CCA sleep profiles with permutation testing accounting for family structure

clear
close all
clc


% PALM package is required for permutation testing accounting for family structure 
addpath('palm');


%% Load data (sleep and BPS)

data1 = readtable('data_sleep_bps.csv');        
confounds_data = readtable('confounds_data.csv');     


%% Get permutation order for sample

addpath(genpath(fullfile(root_dir,'CBIG_private','external_packages','matlab','non_default_packages','palm')));
cd(fullfile(cca_dir,'permutation_order'));

family_structure_file = 'retricted_data.csv''
nPerms = 10000;
[EB, EB_table] = hcp2blocks(family_structure_file, ['EB_N' num2str(size(subj,1)) '.csv'], false, subj); 
subj_new = EB(:,5);
for iter_subj = 1:length(subj_new)
    subj_ids(iter_subj,1) = iter_subj;
end
[Pset, VG] = palm_quickperms(subj_ids, EB, nPerms); %, EE, ISE, CMCx, CMCp);


%% Regress out covariates (age, sex, education)

[data_res, ~, ~, ~] = CBIG_glm_regress_matrix(data1, confounds_data, 0, []);
sleep = data_res(:,sleep_vars);
behav = data_res(:,behav_vars);


%% Run CCA 

[A,B,r,U,V,stats] = canoncorr(sleep, behav);
% A, B : coefficients / U, V : scores

% Compute structural coefficients
structCoeff_sleep = corr(sleep, U);
structCoeff_behav = corr(behav, V);


%% Permutation testing

for iter_perm = 1:nPerms
        
    % Select permutation order
    this_order = Pset(:,iter_perm);
    
    % Change order of sleep matrix
    sleep_perm = sleep(this_order,:);
    
    % Run CCA
    [Ap,~,rp,~,~] = canoncorr(sleep_perm, behav);
    
%     % Match components (re-order) relying on sleep coefficients (A)    
%     costMat_sleep = zeros(size(A,2), size(Ap,2));
%     for rowIdx = 1:size(A,2)
%         for colIdx = 1:size(Ap,2)
%             corrmat = corrcoef(A(:,rowIdx), Ap(:,colIdx));
%             costMat_sleep(rowIdx, colIdx) = 1-abs(corrmat(1, 2));
%         end
%     end
%     
%     % Run the Hungarian matching algorithm
%     [order_sleep, ~] = munkres(costMat_sleep);
%     rp_reorder = rp(order_sleep);
%     
%     clear costMat_sleep rowIdx colIdx corrmat order_sleep Ap rp
%     
%     % Create matrix of R-value in permutations
    r_perm(iter_perm,:) = rp; %rp_reorder;
%     clear this_order sleep_perm rp
    
end

% Compute p-values for each component
for iter_comp = 1:size(r,2)
    pvals(iter_comp,1) = (sum(abs(r_perm(:,iter_comp)) >= abs(r(1,iter_comp))) + 1) / (nPerms + 1) ;
end

% FDR correction over the first x LCs
[signif_comp, ~] = FDR(pvals(1:7), 0.05);

% Sort significant components
signif_comp = sort(signif_comp,'ascend')



%% Bootstrapping

nBootstraps = 10000;
nSubj = size(U,1);

boot_order = rri_boot_order(nSubj,1,nBootstraps);

for iter_boot = 1:nBootstraps
         
    % Change order of sleep matrix
    sleep_boot = sleep(boot_order(:,iter_boot), :);
    behav_boot = behav(boot_order(:,iter_boot),:);
    
    [A_boot,~,~,U_boot,V_boot,~] = canoncorr(sleep_boot, behav_boot);

    % Match components (re-order) based on sleep coefficients (A)    
    costMat_sleep = zeros(size(A,2), size(A_boot,2));
    for rowIdx = 1:size(A,2)
        for colIdx = 1:size(A_boot,2)
            corrmat = corrcoef(A(:,rowIdx), A_boot(:,colIdx));
      
            costMat_sleep(rowIdx, colIdx) = 1-abs(corrmat(1, 2));
        end
    end
    
    % Run the Hungarian matching algorithm
    [order_sleep, ~] = munkres(costMat_sleep);
    
    % Recompute the correlation with sorted components
    corr1_sleep = zeros(size(A,2), 1);
    for iter_lc = 1:size(A,2)
        corrmat1 = corrcoef(A(:,iter_lc), A_boot(:,order_sleep(iter_lc)));
        corr1_sleep(iter_lc) = corrmat1(1, 2);
    end
       
    % Re-order scores if needed
    U_boot_reorder = U_boot(:,order_sleep);
    V_boot_reorder = V_boot(:,order_sleep);
    
    % Invert scores if needed
    for iter_lc = 1:size(A,2)
       if corr1_sleep(iter_lc) < 0
           U_boot_reorder(:,iter_lc) = -U_boot_reorder(:,iter_lc);
           V_boot_reorder(:,iter_lc) = -V_boot_reorder(:,iter_lc);
       end
    end
    
    clear costMat_sleep rowIdx colIdx corrmat corrmat1 corr1_sleep ...
        order_sleep A_boot U_boot V_boot
    
    % Compute structural coefficients and store them separately for each lc
    these_structCoeff_sleep_boot = corr(sleep_boot, U_boot_reorder);
    these_structCoeff_behav_boot = corr(behav_boot, V_boot_reorder);
    
    for iter_lc = 1:size(U,2)
        structCoeff_sleep_boot{iter_lc,1}(:,iter_boot) = these_structCoeff_sleep_boot(:,iter_lc);
        structCoeff_behav_boot{iter_lc,1}(:,iter_boot) = these_structCoeff_behav_boot(:,iter_lc);
    end
    
end


%% Compute standard deviation, z-scores and p.values across bootstrap samples

for iter_lc = 1:size(U,2)
    
    for iter_sleep = 1:size(structCoeff_sleep,1)
        mean_structCoeff_sleep_boot(iter_sleep,iter_lc) = mean(structCoeff_sleep_boot{iter_lc}(iter_sleep,:));
        std_structCoeff_sleep_boot(iter_sleep,iter_lc) = std(structCoeff_sleep_boot{iter_lc}(iter_sleep,:));
        zscore_structCoeff_sleep_boot(iter_sleep,iter_lc) = structCoeff_sleep(iter_sleep,iter_lc) / std_structCoeff_sleep_boot(iter_sleep,iter_lc);
        
        if zscore_structCoeff_sleep_boot(iter_sleep,iter_lc) >= 0
            pvals_structCoeff_sleep_boot(iter_sleep,iter_lc) = 1-cdf('norm',zscore_structCoeff_sleep_boot(iter_sleep,iter_lc),0,1);
        elseif zscore_structCoeff_sleep_boot(iter_sleep,iter_lc) < 0
            pvals_structCoeff_sleep_boot(iter_sleep,iter_lc) = cdf('norm',zscore_structCoeff_sleep_boot(iter_sleep,iter_lc),0,1);
        end
    end
    
    
    for iter_behav = 1:size(structCoeff_behav,1)
        mean_structCoeff_behav_boot(iter_behav,iter_lc) = mean(structCoeff_behav_boot{iter_lc}(iter_behav,:));
        std_structCoeff_behav_boot(iter_behav,iter_lc) = std(structCoeff_behav_boot{iter_lc}(iter_behav,:));
        zscore_structCoeff_behav_boot(iter_behav,iter_lc) = structCoeff_behav(iter_behav,iter_lc) / std_structCoeff_behav_boot(iter_behav,iter_lc);
        
        if zscore_structCoeff_behav_boot(iter_behav,iter_lc) >= 0
            pvals_structCoeff_behav_boot(iter_behav,iter_lc) = 1-cdf('norm',zscore_structCoeff_behav_boot(iter_behav,iter_lc),0,1);
        elseif zscore_structCoeff_behav_boot(iter_behav,iter_lc) < 0
            pvals_structCoeff_behav_boot(iter_behav,iter_lc) = cdf('norm',zscore_structCoeff_behav_boot(iter_behav,iter_lc),0,1);
        end
    end
    
end



%% Compute GLM between CCA loadings & RSFC 


% Load RSFC
these_rsfc = load('rsfc_data.mat');

% Define inverted LCs (note: directionality in CCA is arbitrary)
inverted_lc = [1,2,5,7]; 
   
for iter_lc = 1:size(U,2)
    if any(iter_lc == inverted_lc)
        these_U(:,iter_lc) = -U(:,iter_lc);
        these_V(:,iter_lc) = -V(:,iter_lc);
    end
end

% Average sleep and behavior loadings
for iter_lc = 1:size(U,2)
    these_loadings = [these_U(:,iter_lc) these_V(:,iter_lc)];
    these_mean_loadings(:,iter_lc) = mean(these_loadings,2); 
    clear these_loadings
end


%% Regress out confounds from RSFC

[these_rsfc_reg, ~, ~, ~] = CBIG_glm_regress_matrix(these_rsfc', confounds_data, 0, []);


%% Fit GLM between CCA loadings & RSFC

for iter_lc = 1:size(U,2)
    
    disp(['Running GLM on LC' num2str(iter_lc)]);
    
    x = these_mean_loadings(:,iter_lc);
    y = these_rsfc_reg;
    
    % Generalized linear model
    for iter_edge = 1:size(these_rsfc_reg,2)
        
        mdl = fitglm(x, y(:,iter_edge), 'linear'); % fit model
        
        % Save beta coefficients, standard error & t-val for each edge
        all_betas{iter_lc}(iter_edge,1) = table2array(mdl.Coefficients(2,1));
        all_betas_se{iter_lc}(iter_edge,1) = table2array(mdl.Coefficients(2,2));
        all_betas_t{iter_lc}(iter_edge,1) = table2array(mdl.Coefficients(2,3));
        
        [p,f] = coefTest(mdl); % linear hypothesis test on coefficients to get f-statistic of the model (H0=B==0)
        all_glm_p{iter_lc}(iter_edge,1) = p;
        all_glm_f{iter_lc}(iter_edge,1) = f;
        
        clear mdl p f
    end
    
    clear x y
end