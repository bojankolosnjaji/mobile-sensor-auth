function [y_res, output] = OCSVM_Budget_Test(X,y,alpha,K, X_test, y_test, rho)
    num_test_samples = size(X_test,1);
    num_training_samples = size(X,1);
       
    K_test = zeros(num_test_samples, num_training_samples);
    
    y_res = zeros(num_test_samples);
    
    sigma = 1;
    
    for i=1:num_test_samples
        for j=1:num_training_samples        
            K_test(i,j) = exp(-((X_test(i,:)-X(j,:))*((X_test(i,:)-X(j,:))'))/(2*sigma*sigma));
      
        end
        
    end
    output = K_test*alpha;
    y_res = sign(output-rho);
    y_res(y_res==0) = -1;
    %sum(y_test==y_res)*100/size(y_test,1)

end