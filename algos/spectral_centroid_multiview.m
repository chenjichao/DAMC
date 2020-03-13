function Y = spectral_centroid_multiview(X,num_views,numClust,sigma,lambda,numiter)
% INPUT:
% OUTPUT:

% if (min(truth)==0)
%     truth = truth + 1;
% end

[N M1] = size(X{1});
%[N M2] = size(X2);

for i=1:num_views
    %options(i) = [];
    options(i).KernelType = 'Gaussian';
    options(i).t = sigma(i);
    options(i).d = 4;
end

kmeans_avg_iter = 10;
opts.disp = 0;

numEV = numClust;
numVects = numClust;
for i=1:num_views
% Laplacian for the first view of the data
    K(:,:,i) = constructKernel(X{i},X{i},options(i));
    %K1 = X1*X1';
    D = diag(sum(K(:,:,i),1));
    %L1 = D1 - K1; 
    L(:,:,i) = sqrt(inv(D))*K(:,:,i)*sqrt(inv(D));  
    L(:,:,i)=(L(:,:,i)+L(:,:,i)')/2;
    [U(:,:,i) E] = eigs(L(:,:,i),numEV,'LA',opts);    
    objval(i,1) = sum(diag(E));
end

%%do clustering for first view
U1 = U(:,:,1);
normvect = sqrt(diag(U1*U1'));
normvect(find(normvect==0.0)) = 1;
U1 = inv(diag(normvect)) * U1;    
% for j=1:kmeans_avg_iter
%     C = kmeans(U1(:,1:numVects),numClust,'EmptyAction','drop'); 
%     newnmi_j(j) = calculate_nmi(truth,C);
% end
% newnmi(1) = mean(newnmi_j); std_newnmi(1) = std(newnmi_j);


i = 2;
% now iteratively solve for all U's
while(i<=numiter+1)

    L_ustar(1:N,1:N) = 0;
    for j=1:num_views
        L_ustar = L_ustar + lambda(j)*U(:,:,j)*U(:,:,j)';
    end
    L_ustar = (L_ustar+L_ustar')/2;
    [Ustar, Estar] = eigs(L_ustar, numEV,'LA',opts);    

    L_ustar = Ustar*Ustar';
    L_ustar = (L_ustar+L_ustar')/2;
    for j=1:num_views            
        [U(:,:,j) E] = eigs(L(:,:,j) + lambda(j)*L_ustar, numEV,'LA',opts);    
        objval(j,i) = sum(diag(E));
    end

    objval(1,i) = sum(diag(E));

%     if (1)  %use view 1 in actual clustering %% means view star I think
%         U1 = Ustar;
%         normvect = sqrt(diag(U1*U1'));    
%         normvect(find(normvect==0.0)) = 1;
%         U1 = inv(diag(normvect)) * U1;
% 
%         for j=1:kmeans_avg_iter
%             C = kmeans(U1(:,1:numVects),numClust,'EmptyAction','drop'); 
%             newnmi_j(j) = calculate_nmi(truth,C);
%         end
%         newnmi(i) = mean(newnmi_j); std_newnmi(i) = std(newnmi_j);
%     end
    i = i+1;
end

% fprintf('\n');
% fprintf('newnmi: ');    
% for i=1:numiter+1      
%     fprintf('\n       %f(%f)', newnmi(i), std_newnmi(i));
% end

% fprintf('\n');
% for j=1:num_views
%     fprintf('objval_u%d:   ', j);    
%     for i=1:numiter+1
%         fprintf('%f  ', objval(j,i));
%     end
%     fprintf('\n');
% end


U1 = Ustar;
normvect = sqrt(diag(U1*U1'));    
normvect(find(normvect==0.0)) = 1;
U1 = inv(diag(normvect)) * U1;

Y = kmeans(U1(:,1:numVects),numClust,'EmptyAction','drop'); 
