function labels = GSF(So, c, gamma1, gamma2, k)
% X:                cell array, 1 by view_num, each array is num by d_v
% c:                number of clusters
% v:                number of views
% k:                number of adaptive neighbours
%groundtruth£º      groundtruth of the data, num by 1

% if nargin < 7
%     k = 9;
% end;
islocal = 1;
v = size(So,3);
num = size(So,1);
NITER = 30;

% A = sum(So,3);
% A = (A + A')/2;
% DA = diag(sum(A));
% LA = DA - A;
% [U, ~, ~]=eig1(LA, c, 0);
% S = ones(num);
% for i = 1:v
%     S = S.*So(:,:,i);
% end

S = ones(num);
for i = 1:v
    S = S-diag(diag(S));
    S = S.*So(:,:,i);
end
S = (S + S')/2;
DA = diag(sum(S));
LA = DA - S;
[U, ~, ~]=eig1(LA, c, 0);
Sa = S;
% [clusternum, ~] = graphconncomp(sparse(Sa))
%% =====================   Normalization =====================
% % for i = 1 :v
%     for  j = 1:num
%       X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
%     end
% end
%% =====================  updating =====================
distX = L2_distance_1(U',U');
% distX = L2_distance_1(X{3}',X{3}');
[~,idx] = sort(distX,2);
obj = [];
for iter = 1:NITER
    %update S
    S = zeros(num);
    distu = L2_distance_1(U',U');
    for i=1:num
        if islocal ==1
            idxa0 = idx(i,2:k + 1);
        else
            idxa0 = 1:num;
        end
        dui = distu(i,idxa0);
        dus = Sa(i,idxa0);
        ad = -(dui - gamma1*dus)/(2*gamma2);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    
    S = (S + S')/2;
    D = diag(sum(S));
    L = D - S;
    U_old = U;
    % Update U
    [U, ~, ev]=eig1(L, c, 0);
    thre = 1*10^-5;
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    ob = trace(U'*L*U) - gamma1*trace(S*Sa) + gamma2*trace(S*S') ;
    obj = [obj ob];
    if fn1 > thre
        gamma2 = gamma2/2;
    elseif fn2 < thre
        gamma2 = gamma2*2;  U = U_old;
    else
        break;
    end
%     sprintf('iter = %d',iter)
    
end
[~, y] = graphconncomp(sparse(S));
labels = y;
