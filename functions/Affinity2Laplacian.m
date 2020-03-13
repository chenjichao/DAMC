function L = Affinity2Laplacian(A, normalize)
% A: Adjacency matrix
% D: Degree matrix
% L: Laplacian matrix

if nargin < 2
    normalize = false;
end

D = sum(A, 2);
if normalize
    D(D~=0) = sqrt(1./D(D~=0));
    D = spdiags(D, 0, speye(size(A, 1)));
    A = D*A*D;
    L = speye(size(A, 1)) - A; % L = I-D^-1/2*W*D^-1/2
else
    D = spdiags(D, 0, speye(size(A, 1)));
    L = D - A;
end