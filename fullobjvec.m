function [obj,grad] = fullobjvec(x,p,g,nSig)
%FULLOBJVEC Full Optimal screening objective (vectorised)
%   Detailed explanation goes here

nTypes = length(p);
nVarsMech = 2*nSig*nTypes;
nVarsSig = nSig*nTypes^2;
nVars = nVarsMech + nVarsSig;

mech = x(nVarsSig+1:nVars);
sig = permute(reshape(x(1:nVarsSig),nSig,nTypes,nTypes),[2,1,3]);

[rev,revgrad] = exprevvec(p,mech,sig);
[cost,costgrad] = expentcostvec(p,sig);

obj = -rev + g * cost;
grad = -revgrad + g * costgrad;

end
