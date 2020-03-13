function norm_dists_all = distancefusion_within(X, fusion1, fusion2)
% size(X{iViw}) = [nFtr, nSmp]
% fusion = 'am','AM','arithmetic mean' or
%          'gm','GM','geometric mean' or
%          'hm','HM','harmonic mean' or
%          'no','NO','no fusion'

if nargin<3
    fusion2 = 'AM';
end
if nargin<2
    fusion1 = 'GM';
end

nViw = length(X);
nSmp = size(X{1}, 1);
norm_dists_all = zeros(nViw, nSmp, nSmp);
norm_d = zeros(2, nSmp, nSmp);
for iViw = 1:nViw
    dij = L2_distance_1(X{iViw}');
    di2 = sqrt(sum(abs(dij).^2, 1));
    
    % distance normalization
    switch lower(fusion1)
        case {'no','no fusion'}
            norm_dists_all(iViw,:,:) = dij;
        case {'am','arithmetic mean'}
            d_top = di2'+di2;
            d_btm = di2'*di2;
            norm_dists_all(iViw,:,:) = 0.5*(d_top.*dij)./d_btm;
        case {'hm','harmonic mean'}
            norm_dists_all(iViw,:,:) = squeeze(harmmean(norm_d, 1));
        case {'gm','geometric mean'}
            d_btm = sqrt(di2'*di2);
            norm_dists_all(iViw,:,:) = dij./d_btm;
        otherwise % geometric mean
            d_btm = sqrt(di2'*di2);
            norm_dists_all(iViw,:,:) = dij./d_btm;
    end
end

end % end of function


function d = L2_distance_1(a,b)
% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b

if nargin < 2
    b = a;
end

if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))];
  b = [b; zeros(1,size(b,2))];
end

aa = sum(a.*a);
bb = sum(b.*b);
ab = a'*b;
d = repmat(aa', [1, size(bb, 2)]) + repmat(bb, [size(aa, 2), 1]) - 2*ab;
d = real(d);
d = max(d,0);

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end
end % end of L2_distance_1