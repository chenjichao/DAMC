function y = MCGC(X, nCls, nNbr, beta)

numiter = 10;


%% X to W

islocal = 1;
nSmp = size(X{1}, 1);
nViw  = length(X);
W = zeros(nSmp, nSmp, nViw);
for iViw = 1:nViw 
	W(:,:,iViw) = Updata_Sv(X{iViw}', nCls, nNbr, islocal);
end


%% original
N = size(W, 1);
opts.disp = 0;
U = zeros(N, nCls, nViw);
gamma = 1;
S(1:N, 1:N) = 0;
for v = 1:nViw
    fprintf('computing embedding matrix for view (%d)\n', v);
    [U(:, :, v), ~] = eigs(W(:, :, v), nCls, 'LA', opts);
    S = S + beta*U(:, :, v)*U(:, :, v)';
end

S = (S+S')/2;
DA = diag(sum(S));
LA = DA - S;
[H, ~, ~] = eig1(LA, nCls, 0);
zr = 10e-11;
k = 2;
OBJ = zeros(numiter+1, 1);

for v = 1:nViw
    OBJ(1) = OBJ(1) + trace(U(:, :, v)'*LA*U(:, :, v)) + ...
        norm(S - beta*U(:, :, v)*U(:, :, v)', 'fro');
end

while(k <= numiter+1)
    fprintf('Running iteration %d\n', k-1);   
    A0 = zeros(N);
    for v = 1:nViw            
        [U(:, :, v), ~] = eigs(W(:, :, v) + beta.*S, nCls, 'LA', opts);
        A0 = A0 + beta*U(:, :, v)*U(:, :, v)';
    end
    for iter = 1:50
        dist = L2_distance_1(H', H');
        S = A0.*0;
        for j = 1:N
            ai = A0(j, :);
            di = dist(j, :);
            ad = ai - 0.5.*gamma*di; 
            S(j,:) = EProjSimplex_new(ad);
        end
        S = (S + S.')/2;
        D = diag(sum(S));
        L = D - S;
        F_old = H;
        [H, ~, ev] = eig1(L, nCls, 0);
        fn1 = sum(ev(1:nCls));
        fn2 = sum(ev(1:nCls+1));
        if fn1 > zr
            gamma = gamma.*2;
        elseif fn2 < zr
            gamma = gamma/2;  
            H = F_old;
        else
            break;
        end
    end
    for v = 1:nViw
        OBJ(k) = OBJ(k) + trace(U(:, :, v)'*L*U(:, :, v)) + ...
            norm(S-beta*U(:, :, v)*U(:, :, v)', 'fro');
    end
    k = k+1;
end


% plot(OBJ)
[~, y] = graphconncomp(sparse(S));
y = y';
% [acc, nmi, Pu] = ClusteringMeasure(truth, y);
% AR = RandIndex(truth, y+1);
% [F,P,R] = compute_f(truth,y);