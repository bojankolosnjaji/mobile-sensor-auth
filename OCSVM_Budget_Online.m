function [ X_all,y,alpha_new,K_new, output ] = OCSVM_Budget_Online( X, X_new, K, alpha, params, strategy)
%OCSVM_Budget_Online Implementation of the One-Class SVM with a fixed limit on
%model size - number of support vectors
%   X - inputs
%   params
%   strategy: 0-no strategy, 1-deleting, 2-projection, 3-merging

nu = params(1);
sigma = params(2);
lambda = params(3);
budget = params(4);
l = size(X,1);

if (~isempty(K))
    K_new = zeros(l+1,l+1);
    K_new(1:l,1:l) = K;
    for i=1:l                
        K_new(i,l+1) = exp(-((X(i,:)-X_new)*((X(i,:)-X_new)'))/(2*sigma*sigma));
        K_new(l+1,i) =  K_new(i,l+1);
    end
    K_new(l+1,l+1)=1;
else
    K_new = 1;
end

a = 0;
b = min([1,1/(nu*(l+1))]);
alpha_addition = a + (b-a).*rand(1,1);
alpha_new = [alpha ;alpha_addition]; % at first, we compute a random alpha and add it to the old vector
grad_alpha = (1/2)*(K_new*alpha_new); % then we need a gradient

gradient_steps = 100;

for i=1:gradient_steps
    alpha_new = alpha_new-grad_alpha*lambda;
end
% find a minimal support vector

[min_alpha, ind_min_alpha] = min(alpha_new);

if (sum(alpha_new~=0)>=budget)
  %  ['rationalizing']
    if (strategy==1) % deletion
        alpha_new(ind_min_alpha)=0;

    elseif (strategy==2) % projection
        K_p = [K_new(1:ind_min_alpha-1,:); K_new(ind_min_alpha+1:size(K_new,0),:)];
        k_p = K_p(:,ind_min_alpha);
        K_p = [K_p(:,1:ind_min_alpha-1), K_p(:,ind_min_alpha+1:size(K_new,1))];

        alpha_new = alpha_new(ind_min_alpha) * (inv(K_p)) * (k_p');
        alpha_new(ind_min_alpha)=0;      
    elseif(strategy==3) % merging

    else % no strategy

    end
end
alpha_new = alpha_new/sum(alpha_new,1);
output = K_new*alpha_new;
y = sign(output-0.01);
X_all = [X;X_new];


end

