function labels = MLAN(X,c,groundtruth, nNbr)
% X:                cell array, 1 by view_num, each array is num by d_v
% c:                number of clusters
% v:                number of views
% k:                number of adaptive neighbours
%groundtruth£º      groundtruth of the data, num by 1

% if nargin < 4
%     k = 9;
% end

k = nNbr;

v = size(X,2);
num = size(X{1},1);
lambda = randperm(30,1);
NITER = 30;


%% =====================   Normalization =====================
% for i = 1 :v
%     for  j = 1:num
%         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
%     end
% end


%% =====================  Initialization =====================

%initialize weighted_distX
SUM = zeros(num);
for i = 1:v
    distX_initial(:,:,i) =  L2_distance_1( X{i}',X{i}' );                  %initialize X

    SUM = SUM + distX_initial(:,:,i);
end
distX = 1/v*SUM;
[distXs, idx] = sort(distX,2);

%initialize S
S = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distXs(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);               %initialize S
end
alpha = mean(rr);

% initialize F
S = (S+S')/2;                                                         % initialize F
D = diag(sum(S));
L = D - S;
[F, temp, evs]=eig1(L, c, 0);

if sum(evs(1:c+1)) < 0.00000000001
    error('The original graph has more than %d connected component', c);
end


%% =====================  updating =====================

for iter = 1:NITER
    % update weighted_distX
    SUM = zeros(num,num);
    for i = 1 : v
        if iter ==1
            distX_updated = distX_initial;
        end
        Wv(i) = 0.5/sqrt(sum(sum( distX_updated(:,:,i).*S)));    % update X
        distX_updated(:,:,i) = Wv(i)*distX_updated(:,:,i) ;
        SUM = SUM + distX_updated(:,:,i);
    end
    distX = SUM;
    
    %update S
    distf = L2_distance_1(F',F');
    S = zeros(num);
    for i=1:num                                                  % update A
        idxa0 = idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    
    %update F
    S = (S+S')/2;                                   
    D = diag(sum(S));
    L = D-S;
    F_old = F;
    [F, ~, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;
    
    %update lambda
    thre = 1*10^-10;
    fn1 = sum(ev(1:c));                                     % update lambda
    fn2 = sum(ev(1:c+1));
    if fn1 > thre
        lambda = 2*lambda;
    elseif fn2 < thre
        lambda = lambda/2;  F = F_old;
    else
        break;
    end
%     fprintf('\niter = %d', iter);
end

%% =====================  result =====================
[clusternum, y]=graphconncomp(sparse(S)); y = y';
if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', cluster_num);
    labels = [];
end
% result = ClusteringMeasure(groundtruth, y);
labels = y;