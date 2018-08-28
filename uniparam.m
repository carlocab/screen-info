function [p,v,surplus] = uniparam(nTypes,varargin)
%UNIPARAM Uniformly distributed parameters
%   Detailed explanation goes here

% Generate a uniformly distributed probability distribution
p = exprnd(1,nTypes,1);
p = p / sum(p);

% Generate a uniformly distributed distribution of values
v = rand(nTypes,1);
v = sort(v,'descend');

if ~isempty(varargin)
    max_v = varargin{1};
    v = max_v * v;
end

surplus = p'*v;

end

