function [c,ceq,gradc,gradceq] = mechconstrvec(v,mech,sig)
%MECHCONSTR Incentive and participation constraints (vectorised)
%   MECHCONSTR returns the values of the incentive and participation
%   constraints and their gradients.

[nTypes,nSig,~] = size(sig);
nVarsMech = 2*nSig*nTypes;
nVarsSig = nSig*nTypes^2;
nVars = nVarsMech + nVarsSig;

IR = zeros(nTypes,nVarsMech);
IC = zeros(nTypes*(nTypes-1),nVarsMech);
IRgrad = zeros(nTypes,nVars);
ICgrad = zeros(nTypes*(nTypes-1),nVars);

% Build IR and IC constraints

% IR constraint indexing
i = repelem((1:nTypes)',nSig); % True type counter
j = (1:(nVarsMech/2))'; % Type-signal pair counter
k = repmat((1:nSig)',nTypes,1); % Signal counter
IRsub = sub2ind([nTypes,nVarsMech],i,j); % Linear indexing for IR constraint matrix
sigsub = sub2ind([nTypes,nSig,nTypes],i,k,i); % Linear indexing for signal array

% IR constraints
IR(IRsub) = sig(sigsub);
IR(IRsub + nVarsSig) = -v(i) .* sig(sigsub);

% IC constraint indexing
i = repelem((1:nTypes*(nTypes-1))',nSig); % IC constraint counter
j = reshape(1:nSig*nTypes,nSig,nTypes); % Type-signal pair counter for assignment into columns of IC
j = repelem(j,1,nTypes-1);
k = repelem((1:nTypes)',(nTypes-1)*nSig); % True type counter
ii = repmat((1:nSig)',nTypes*(nTypes-1),1); % Signal counter
ICsub = sub2ind([nTypes*(nTypes-1),nVarsMech],i,j(:));
sigsub = sub2ind([nTypes,nSig,nTypes],k,ii,k);

% IC constraints
IC(ICsub) = sig(sigsub);
ICsub = sub2ind([nTypes*(nTypes-1),nVarsMech],i,j(:) + nTypes*nSig);
IC(ICsub) = -v(k) .* sig(sigsub);

% More IC indexing
m = cycperm(1:nTypes);
m = m(2:end,:);
m = repelem(m(:),nSig); % Announced type counter
j = cycperm(1:nTypes*nSig);
j = j((1+nSig):end,1:nSig:end); % New (announced-)type-signal pair counter
ICsub = sub2ind([nTypes*(nTypes-1),nVarsMech],i,j(:));
sigsub = sub2ind([nTypes,nSig,nTypes],m,ii,k);

% More IC constraints
IC(ICsub) = -sig(sigsub);
ICsub = sub2ind([nTypes*(nTypes-1),nVarsMech],i,j(:) + nTypes * nSig);
IC(ICsub) = v(k).* sig(sigsub);

% Build IR and IC constraint gradients
i = (1:nTypes)';
j = nVarsSig+1:nVars;
IRgrad(i,j) = IR;
i = (1:nTypes*(nTypes-1))';
ICgrad(i,j) = IC; % passed

% Indexing for IR gradient wrt signals
i = 1:nTypes; % Type/IR constraint counter
j = (1:nSig)' + (i-1)*nSig*(nTypes+1); % Type-signal pair counter
i = repelem(i',nSig);
IRsub = sub2ind([nTypes,nVars],i,j(:)); % Linear indexing for IR gradient
mechsub = 1:nSig*nTypes; % Linear indexing for mechanism

% IR gradient wrt signals
IRgrad(IRsub) = mech(mechsub) - v(i) .* mech(mechsub + nTypes*nSig);

% Indexing for IC gradient wrt signals
i = 1:nTypes;
j = (1:nSig)' + (i-1)*nSig*(nTypes+1); % Type-signal pair counter
j = repelem(j,1,nTypes-1);
i = repelem((1:nTypes*(nTypes-1))',nSig);  % IC constraint counter
mechind = repelem(reshape(1:nTypes*nSig,nSig,nTypes),1,nTypes-1); % Mechanism indexing
typeind = repelem(1:nTypes,nSig*(nTypes-1)); % Type indexing
ICsub = sub2ind(size(ICgrad),i,j(:)); % Linear indexing for IC gradient

ICgrad(ICsub) = mech(mechind(:)) - v(typeind) .* mech(mechind(:) + nSig*nTypes); % passed

j = reshape(1:nSig*(nTypes+1)*(nTypes-1),nSig*(nTypes+1),(nTypes-1))';
j = j(:,(nSig+1):end);
j = reshape(j',nSig,numel(j)/nSig);

for ii = 1:(nTypes-1)
    jcolind = ii*(nTypes-1)+1:(ii+1)*(nTypes-1);
    j(:,jcolind) = circshift(j(:,jcolind),nTypes-1-ii,2);
end

ICsub = sub2ind(size(ICgrad),i,j(:)); % Linear indexing for IC gradient
mechind = cycperm(1:nTypes*nSig); 
mechind = mechind(1:nSig:end,nSig+1:end)'; % Mechanism indexing
ICgrad(ICsub) = -mech(mechind(:)) + v(typeind) .* mech(mechind(:) + nSig*nTypes);

A = [IR;IC];
c = A * mech;
gradc = [IRgrad;ICgrad]';
ceq = [];
gradceq = [];
end
