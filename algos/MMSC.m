function labels = MMSC(X, nCls, alpha, nNbr)
nViw = length(X);
Ln = [];
for iViw = 1:nViw
    X{iViw} = normalize(X{iViw});
    [A, ~] = selftuning(X{iViw}, nNbr);
    Ln(:,:,iViw) = Adjacency2Laplacian(A);
end
labels = multimodal_spectral_clustering(Ln, nCls, alpha, 'kmeans');
[~, labels] = max(labels, [], 2);