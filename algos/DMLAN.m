function labels = DMLAN(X, nCls, nNbr, alpha, nOut, fusion3)
% X:                cell array, 1 by view_num, each array is num by d_v
% c:                number of clusters
% v:                number of views
% k:                number of adaptive neighbours
%groundtruth£º      groundtruth of the data, num by 1

% if nargin < 4
%     k = 9;
% end

k = nNbr;
nViw = size(X,2);
nSmp = size(X{1},1);
nItr = 30;
lambda = randperm(30,1);


%% =====================   Normalization =====================
% for i = 1 :nViw
%     for  j = 1:nSmp
%         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
%     end
% end


%% =====================  Initialization =====================

%initialize weighted_distX
SUM = zeros(nSmp);
for i = 1:nViw
    distX_updated(:,:,i) =  L2_distance_1( X{i}',X{i}' );                  %initialize X

    SUM = SUM + distX_updated(:,:,i);
end
distX = 1/nViw*SUM;
[distXs, idx] = sort(distX,2);

%initialize S
S = zeros(nSmp);
rr = zeros(nSmp,1);
for i = 1:nSmp
    di = distXs(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);               %initialize S
end
rr = mean(rr);

% initialize F
S = (S+S')/2;                                                         % initialize F
D = diag(sum(S));
L = D - S;
[F, ~, evs]=eig1(L, nCls, 0);

if sum(evs(1:nCls+1)) < 0.00000000001
    error('The original graph has more than %d connected component', nCls);
end

% init ELM embedding
nNrn = 1000;
distE = zeros(nSmp,nSmp,nViw);
for iViw = 1:nViw
    W{iViw} = rand(size(X{iViw},2), nNrn)*2-1; % W is random input weights
    H{iViw} = X{iViw}*W{iViw};
%     b{iViw} = rand(1,nNrn);
%     H{iViw} = H{iViw} + b{iViw};
    H{iViw} = 1./(1+exp(-H{iViw}));
end

%% =====================  updating =====================
for iter = 1:nItr
    % update Embedding
    for iViw = 1:nViw
        X = H{iViw}';
        if nNrn < nSmp
            X_center = X-repmat(mean(X,2),1,size(X,2));
            AA = alpha*eye(nNrn)+X*L*X';
            BB = X_center*X_center'+1e-10*eye(size(X,1));
            try
                [E,~] = eigs(AA,BB,nOut,'sm');
            catch
                warning('Cant eigs.');
%                 E = rand(nNrn,nOut);
                cant_eigs = 1;
                break
            end
            norm_term = X_center'*E;
            W{iViw} = bsxfun(@times,E,sqrt(1./sum(norm_term.*norm_term)));
        else
            X_center = X-repmat(mean(X,2),1,size(X,2));
            AA = alpha*eye(nSmp)+L*(X'*X);
            BB = pinv(X'*X)*X'*(X_center*X_center')*X+1e-10*eye(size(X,2));
            try
                [E,~] = eigs(AA,BB,nOut,'sm');
            catch
                warning('Cant eigs.');
%                 E = rand(nNrn,nOut);
                cant_eigs = 1;
                break
            end
            norm_term = X_center'*X*E;
            W{iViw} = bsxfun(@times,X*E,sqrt(1./sum(norm_term.*norm_term)));
        end
        temp{iViw} = H{iViw}*W{iViw};
        distE(:,:,iViw) = L2_distance_1(temp{iViw}',temp{iViw}');                  %initialize X
    end
    
    % update weighted_distDE
    SUM = zeros(nSmp,nSmp);
    for i = 1 : nViw
        switch fusion3
            case 'pd'
                distDE = distX_updated(:,:,i).*distE(:,:,i);
            case 'gm'
                distDE = sqrt(distX_updated(:,:,i).*distE(:,:,i));
            case 'sm'
                distDE = distX_updated(:,:,i)+distE(:,:,i);
            case 'am'
                distDE = (distX_updated(:,:,i)+distE(:,:,i))/2;
            otherwise
                distDE = distX_updated(:,:,i).*distE(:,:,i);
        end
        Wv(i) = 0.5/sqrt(sum(sum( distDE.*S)));    % update weight_view
        distDE = Wv(i)*distDE;
        SUM = SUM + squeeze(distDE);
    end
    distDE = SUM;
    
    %update S
    distf = L2_distance_1(F',F');
    S = zeros(nSmp);
    for i=1:nSmp                                                  % update A
        idxa0 = idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dxi = distDE(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*rr);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    
    %update F
    S = (S+S')/2;                                   
    D = diag(sum(S));
    L = D-S;
    F_old = F;
    [F, ~, ev]=eig1(L, nCls, 0);
    evs(:,iter+1) = ev;
    
    %update lambda
    thre = 1*10^-10;
    fn1 = sum(ev(1:nCls));                                     % update lambda
    fn2 = sum(ev(1:nCls+1));
    if fn1 > thre
        lambda = 2*lambda;
    elseif fn2 < thre
        lambda = lambda/2;
        F = F_old;
    else
        break;
    end
%     fprintf('\niter = %d', iter);
end

iter

%% =====================  result =====================
[clusternum, y]=graphconncomp(sparse(S)); y = y';
if clusternum ~= nCls
    sprintf('Can not find the correct cluster number: %d', clusternum);
    labels = [];
end
% result = ClusteringMeasure(groundtruth, y);
labels = y;