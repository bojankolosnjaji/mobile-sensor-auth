%% Test for various value of Rho (anomaly threshold), for testing keyboard and swipe features together  %%
rng('shuffle')

my_epsilon = 1e-6;

mos=[2,4];
mo = 4;
    
path = './data/';

if(mo==1)
path = strcat(path,'Swipes/');
rest=1:59;
sz=59;
elseif(mo==2)
path = strcat(path,'Write/');
rest=1:51;
sz=51;
sf = [29 32 35	28 34 31 30	33 36 13 19]; 
elseif(mo==3)
path = strcat(path,'Raw/');
rest=1:59;
sz=59;
elseif(mo==4)
path = strcat(path,'Combined/');
rest=1:59;
sz=59;
sf = [37 4 17 36 1 55 21 18 16 28 26 42 46 38 20 43 6 51 35 3 22 33 19 40 23 44];
end

sf_keyboard = [29 32 35	28 34 31 30 33 36 13 19]; 
sf_swipe = [37 4 17 36 1 55 21 18 16 28 26 42 46 38 20 43 6 51 35 3 22 33 19 40 23 44];

Bs = [25 50 75 100 125 150 175 200 250 300 350];
numU = 1;%9;                                   % try for best also
numR = 28; % Number of users?
gamma = 0.1;
% result = struct();

sz = size(Bs,2);
B = 200;

%% PA %%

frr = zeros(numR,sz);
far = zeros(numR,sz);
acc = zeros(numR,sz);
numSV = zeros(numR,sz);
ts = zeros(numR,sz);
ns = zeros(numR,sz);
yps = cell(numR,sz);


t=1;
for l=1:1%numR
if(mo==1)
[predictors, response, predictorsTest, responseTest, ind, n]=getData_AD(l,numU, [], [], [],6,2);
end
if(mo==2)
[predictors, response, predictorsTest, responseTest, ind, ns(l,t)]=getDataK(l,numU, sf, [], 300, [],1);
end
if(mo==3)
[predictors, response, predictorsTest, responseTest, ind, n]=getDataR(l,numU, [], [], 300, 0);
end
if(mo==4)
[predictors, response, predictorsTest, responseTest, ind, ns(l,t)]=getDataRK_AD(l,numU, sf, [], 300, 0, 2,1,8);
end


budgets = [5,10, 20, 50, 100, 150,200];
%budgets = 100
num_tests = 5;
accuracy_budgets = zeros(size(budgets,2));
fa_budgets = zeros(size(budgets,2),1);
fr_budgets = zeros(size(budgets,2),1);
tr_budgets = zeros(size(budgets,2),1);
ta_budgets = zeros(size(budgets,2),1);
acc_budgets = zeros(size(budgets,2),1);
% rho - default 0.00001
rho_values = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
rho_cells = cell(9,1);
for rho_it_swipe = 1:size(rho_values,2)
 %   [rho_it_swipe]
    rho_swipe_cell = cell(9,1);
    for rho_it_keyboard = 1:size(rho_values,2)
        %[rho_values(rho_it_swipe) rho_values(rho_it_keyboard)]
        fa_users = zeros(28,1);
        fr_users = zeros(28,1);
        ta_users = zeros(28,1);
        tr_users = zeros(28,1);
        acc_users = zeros(28,1);
        confusion_matrix = zeros(28,28);
        active_alphas = cell(28,28);

        for valid_user=1:28
            correct_test = 0;
            all_test = 0;
            for attack_user = 1:28
                [valid_user, attack_user]
                test_values = zeros(num_tests,1);
                for test_no = 1:num_tests

                    [predictors, response, predictorsTest, responseTest, ind]=getDataRK_AD(l,numU, sf_swipe, [], 300, 0, 2,valid_user,attack_user);

                    szVa = sum(response==1); 
                    szAt = sum(response==-1); 
                    % online learning test
                    X = [];
                    K = [];
                    alpha = [];
                    params = [ 0.5, 1 0.00001, 300  ];
                    strategy = 1;
                    for i=1:szVa % simulate online arriving samples
                        X_new = predictors(i,:);
                        [ X,y,alpha,K, output] = OCSVM_Budget_Online_Compressed( X, X_new, K, alpha, params, strategy); % training
                    end

                    % testing

                    [y_res_test, output] = OCSVM_Budget_Test(X,y,alpha,K, predictorsTest,responseTest, rho_values(rho_it_swipe));
                    %sum(y_res_test==responseTest)/size(responseTest,1)
                    %sum(alpha~=0)

                     %%%%%%%%%%%%
                    % KEYBOARD %
                    %%%%%%%%%%%%
                    [predictorsK, responseK, predictorsTestK, responseTestK, indK]=getDataK_AD(l,numU, sf_keyboard, [], 300, [], 2,valid_user,attack_user);

                    szVaK = sum(responseK==1); 
                    szAtK = sum(responseK==-1);
                    % online learning test
                    XK = [];
                    K_K = [];
                    alphaK = [];
                    %params = [ 0.5, 1 0.00001, budgets(budget_it)  ];
                    strategy = 5;
                    for i=1:szVaK % simulate online arriving samples
                        X_newK = predictorsK(i,:);
                        [ XK,yK,alphaK,K, outputK] = OCSVM_Budget_Online_Compressed( XK, X_newK, K_K, alphaK, params, strategy); % training
                    end
                    % testing
                    [y_res_testK, outputK] = OCSVM_Budget_Test(XK,yK,alphaK,K_K, predictorsTestK,responseTestK, rho_values(rho_it_keyboard));

                    

                    test_values(test_no) = (sum(y_res_test==responseTest) + sum(y_res_testK == responseTestK))/(size(responseTest,1)+size(responseTestK,1));
                    if (valid_user~=attack_user)
                        for test_in = 1:size(responseTest,1)
                            if (y_res_test(test_in) == responseTest(test_in))
                                correct_test=correct_test+1;
                            end
                            if (y_res_test(test_in) ==1)
                                if (responseTest(test_in)~=1)
                                    fr_users(valid_user) = fr_users(valid_user)+1;
                                else
                                    ta_users(valid_user) = ta_users(valid_user) +1;
                                end
                            else
                                if (responseTest(test_in)==1)
                                    fa_users(valid_user) = fa_users(valid_user)+1;
                                else
                                    tr_users(valid_user) = tr_users(valid_user)+1;
                                end
                            end


                            all_test = all_test +1;
                            
                            

                        end 
                        
   
                         for test_in = 1:size(responseTestK,1)
                            if (y_res_testK(test_in) == responseTestK(test_in))
                                correct_test=correct_test+1;
                            end
                            if (y_res_testK(test_in) ==1)
                                if (responseTestK(test_in)~=1)
                                    fr_users(valid_user) = fr_users(valid_user)+1;
                                else
                                    ta_users(valid_user) = ta_users(valid_user) +1;
                                end
                            else
                                if (responseTestK(test_in)==1)
                                    fa_users(valid_user) = fa_users(valid_user)+1;
                                else
                                    tr_users(valid_user) = tr_users(valid_user)+1;
                                end
                            end


                            all_test = all_test +1;
                         end  
       
                    end

                end

                confusion_matrix(valid_user,attack_user)= mean(test_values);
            end
            acc_users(valid_user) = correct_test/all_test;
        end
        %save('alphas.mat', 'active_alphas')
     %   save(sprintf('out_conf_mat_%d', budgets(budget_it)),'confusion_matrix')
        %save(sprintf('user_results_rho__all_%d_%d.mat', rho_values(rho_it_swipe), rho_values(rho_it_keyboard)), 'fa_users', 'fr_users', 'tr_users', 'ta_users', 'acc_users')
        rho_swipe_cell{rho_it_keyboard} = [fa_users, fr_users, tr_users, ta_users, acc_users];
        
   
    end
    rho_cells{rho_it_swipe} = rho_swipe_cell;
    
end
end

