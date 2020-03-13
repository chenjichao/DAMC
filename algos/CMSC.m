function labels = Coreg(X, nCls, lambda)

nViw = length(X);
nItr = 10;

for iViw = 1:nViw
    sigma(iViw) = optSigma(eval(sprintf('X{%d}', iViw)));
end

% lambda = 0.5;
lambdas = repmat(lambda,1,nViw);

labels = spectral_centroid_multiview(X, nViw, nCls, sigma, lambdas, nItr);
