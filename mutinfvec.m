function [MI,MIgrad] = mutinfvec(p,sig)
%MUTINFVEC Mutual information of p and sig. (vectorised)
%   MUTINFVEC calculates the mutual information of a prior probability p
%   and and a signal structure sig. p is a probability distribution such
%   that p(i) = Pr(i). sig is a matrix of conditional probabilities
%   satisfying sig(i,j) = Pr(j|i). The number of elements of p must be the
%   same as the number of rows in sig, otherwise MATLAB will return an
%   error.
%
%   MUTINFVEC can also accommodate an array of sig matrices to calculate
%   the mutual information of p and different signal structures.
%
%   For the definition of mutual information, see, for example, Cover and
%   Thomas (2006).

[nRow,nCol,nPages] = size(sig); % Track size for reshaping below
sigprob = sum(p(:) .* sig); % (:) indexing ensures p(:) is a column vector
temp = p(:).*sig.*log(sig./sigprob);
temp = reshape(temp,nRow*nCol,nPages);
temp(isnan(temp)) = 0;
MI = sum(temp);

MIgrad = p(:) .* log(sig./sigprob);

end
