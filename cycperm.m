function cperm = cycperm(x)
%CYCPERM Cyclical permutations of vector x
%   Detailed explanation goes here
n = numel(x);
cpind = mod(((0:(n-1))' + (1:n)) - 1,n) + 1;
cperm = x(cpind);
end

