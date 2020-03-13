function labels = ELMAN(X, nCls, nNbr, fusion1, fusion2, alpha, nOut)
% INPUT
% X: num*dim data matrix, each row is a data point
% nCls: number of clusters
% nNbr: number of neighbors to determine the initial graph, 
%       and the parameter r if r<=0
% r: paremeter, which could be set bo a large enough value.
%       If r<0, then it is determined by algorithm with k
% islocal:
%           1: only update the similarities of the k neighbor pairs, the neighbor pairs are determined by the distances in the original space
%           0: update all the similarities
islocal = 1;
% OUTPUT
% y: num*1 cluster indicator vector
original_X = X;

nItr = 30;
nViw = length(X);
nSmp = size(X{1}, 1);
k = nNbr;
bymean0ordistsstar1 = 1;


%% Distance Normalization and Distance Fusion
norm_dists_star = distancefusion(X, fusion1, fusion2); % fused distance matrix

% compute the Unified Affinity A
if bymean0ordistsstar1
    % knn graph, where k=nNbr
    [norm_dists_star_nNbr, idx] = sort(norm_dists_star,2);
    A_star = zeros(nSmp);
    rr = zeros(nSmp,1);
    for i = 1:nSmp
        di = norm_dists_star_nNbr(i,2:k+1);
        rr(i) = 0.5*(k*di(k)-sum(di(1:k)));
        id = idx(i,2:k+1);
        % compute Affinity(similarity) matrix
        A_star(i, id) = (di(k)-di)/(k*di(k)-sum(di(1:k))+eps);
    end
    A_star = (A_star + A_star')/2;
    r = mean(rr);
else
%     A_star = squeeze(mean(A_all, 1)); % not reached
%     r = 1e10; % r: paremeter, which could be set bo a large enough value.
end
lambda = r;



%% Init ELM
nNrn = 1000; % TODO: nNrn as input argument
for iViw = 1:nViw
    W{iViw} = rand(size(X{iViw},2), nNrn)*2-1; % W is random input weights
    H{iViw} = X{iViw} * W{iViw};
%     b{iViw} = rand(1,nNrn);
%     H{iViw} = H{iViw} + b{iViw};
    H{iViw} = 1./(1+exp(-H{iViw}));
end



%% alternating optimization F and A
for iItr = 1:nItr
    
    
    
    
    
%     A_old = A_star;
%     D = diag(sum(A));
%     L = D-A;
    
    
%     fprintf('Iteration: %d/%d', iItr, nItr);
    A_star_old = A_star;
    

    
    % update F with fixed A_star
    L_star = Affinity2Laplacian(A_star);
    [F, ~, ~] = eig1(L_star, nCls, 0);
    distf = L2_distance_1(F',F');
    
    
    
    L = L_star;
    d = 100;
    alpha = 100; % TODO: as a input
    
    % Update embedding
    for iViw = 1:nViw
        X = H{iViw}';
        if nNrn < nSmp
            X_center = X-repmat(mean(X,2),1,size(X,2));
            AA=alpha*eye(nNrn)+X*L*X';
            BB=X_center*X_center'+1e-10*eye(size(X,1));
            [E,~] = eigs(AA,BB,d,'sm');
            norm_term=X_center'*E;
            W{iViw} = bsxfun(@times,E,sqrt(1./sum(norm_term.*norm_term)));
        else
            X_center = X-repmat(mean(X,2),1,size(X,2));
            AA=alpha*eye(nSmp)+L*(X'*X);
            BB=pinv(X'*X)*X'*(X_center*X_center')*X+1e-10*eye(size(X,2));
            [E,~] = eigs(AA,BB,d,'sm');
            norm_term=X_center'*X*E;
            W{iViw} = bsxfun(@times,X*E,sqrt(1./sum(norm_term.*norm_term)));
        end
    end
% d = eigs(A,B,___) solves the generalized eigenvalue problem A*V = B*V*D.

    % Update W
    for iViw=1:nViw
        temp{iViw} = H{iViw}*W{iViw};
    end
    norm_embed_star = distancefusion(temp, fusion1, fusion2); % fused distance matrix
    
    norm_DE_star = norm_dists_star.*norm_embed_star;
    
    % update A_star with fixed F
    % r(lambda) is determined by dist_star for unified A
    A_star = zeros(nSmp);
    for i = 1:nSmp
        if islocal
            idxa0 = idx(i, 2:nNbr+1);
        else
            idxa0 = 1:nSmp;
        end
        
        dfi = distf(i,idxa0);
        dxi = norm_DE_star(i,idxa0);
        ad = -(dxi + lambda*dfi)/(2*r);  % min(s) ||s + d/2r||^2_2
        A_star(i,idxa0) = EProjSimplex_new(ad);
    end
    A_star = (A_star + A_star')/2;
    L_star = Affinity2Laplacian(A_star);
    

    % Check rank to decide whether to continue, repeat or stop
    [~, ~, ev] = eig1(L_star, nCls, 0);
    evs(:, iItr+1) = ev;
    fn1 = sum(ev(1:nCls));
    fn2 = sum(ev(1:nCls+1));
    if fn1 > 1e-11
        lambda = 2*lambda;
    elseif fn2 < 1e-11
        lambda = lambda/2;
        A_star = A_star_old;
    else
        break
    end
    
    norm_dists_star = distancefusion_weighted(original_X, fusion1, fusion2, A_star);
end

%% final results
error = 0;
[clusternum, y] = graphconncomp(sparse(A_star));
labels = y';
if clusternum ~= nCls
    sprintf('Can not find the correct cluster number: %d', nCls)
    error = 1;
    labels = [];
end
