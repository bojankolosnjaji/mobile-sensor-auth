%% Budget size %%
rng('shuffle')

my_epsilon = 1e-6;

mos=[2,4];
mo = 2;
    
path = '/home/kolosnjaji/teaching/theses/BA_Continuous_Authentication/Matlab/Results/Budget/';

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
num_tests = 1;
accuracy_budgets = zeros(size(budgets,2));
fa_budgets = zeros(size(budgets,2),1);
fr_budgets = zeros(size(budgets,2),1);
tr_budgets = zeros(size(budgets,2),1);
ta_budgets = zeros(size(budgets,2),1);
acc_budgets = zeros(size(budgets,2),1);


for budget_it = 1:1
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
            test_values = zeros(num_tests,1);
            for test_no = 1:num_tests
                
                [predictors, response, predictorsTest, responseTest, ind, ns(l,t)]=getDataK_AD(l,numU, sf, [], 300, [], 2,valid_user,attack_user);
                szVa = sum(response==1); 
                szAt = sum(response==-1); 
                % online learning test
                X = [];
                K = [];
                alpha = [];
                params = [ 0.5, 1 0.00001, 300  ];
                strategy = 1;
                active_alpha = zeros(szVa,1);
                for i=1:szVa % simulate online arriving samples
                    X_new = predictors(i,:);
                    [ X,y,alpha,K, output] = OCSVM_Budget_Online_Compressed( X, X_new, K, alpha, params, strategy); % training
                    active_alpha(i) = sum(alpha>my_epsilon);
                end

                % testing

                [y_res_test, output] = OCSVM_Budget_Test(X,y,alpha,K, predictorsTest,responseTest,0.01);
                %sum(y_res_test==responseTest)/size(responseTest,1)
                %sum(alpha~=0)
                test_values(test_no) = sum(y_res_test==responseTest)/size(responseTest,1);
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
                        
            end
            
            confusion_matrix(valid_user,attack_user)= mean(test_values);
            active_alphas{valid_user,attack_user} = active_alpha;
        end
        acc_users(valid_user) = correct_test/all_test;
    end
    save('alphas_keyboard.mat', 'active_alphas')
 %   save(sprintf('out_conf_mat_%d', budgets(budget_it)),'confusion_matrix')
    %save(sprintf('user_results_%d.mat', budgets(budget_it)), 'fa_users', 'fr_users', 'tr_users', 'ta_users', 'acc_users')
end
  
end

