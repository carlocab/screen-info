function [revenue,revgrad] = exprevvec(p,mech,sig)
%EXPREVVEC Seller expected revenue from mech and sig (vectorised)
%   Detailed explanation goes here

nTypes = length(p);
[~,nSig,~] = size(sig);
nVarsMech = 2*nSig*nTypes;
nVarsSig = nSig*nTypes^2;
nVars = nVarsMech + nVarsSig;
revgrad = zeros(nVars,1);
f = zeros(nVarsMech,1);

i = ((1:nTypes)-1)*nSig + (1:nSig)'; % Implicitly calls bsxfun. Compatible with R2016b or later only.
typeind = repmat(1:nTypes,nSig,1);
sigsub = repmat((1:nSig)',1,nTypes);
sigind = sub2ind(size(sig),typeind,sigsub,typeind);
f(i) = p(typeind) .* sig(sigind);

i = ((1:nTypes)-1)*nSig*(nTypes + 1) + (1:nSig)';
mechind = reshape(1:nSig*nTypes,nSig,nTypes);
revgrad(i) = p(typeind) .* mech(mechind);

i = reshape((nVarsSig+1):nVarsSig+nSig*nTypes,nSig,nTypes);
revgrad(i) = p(typeind) .* sig(sigind); % This can probably be simplified with a reference to f(i)

revenue = f' * mech;
end

