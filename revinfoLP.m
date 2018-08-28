function [rev,mech,LM,flag,output] = revinfoLP(p,v,T,S,sig)
%REVINFOLP Optimal trading given information structure
%   Detailed explanation goes here.

% sig(i,j,k): i is the announced type, j is the signal, k is the true type

% Initialise variables
nTypes = length(p);
[~,nSig,~] = size(sig);
nvar = 2*nSig*nTypes;
f = zeros(nvar,1);
IR = zeros(nTypes,nvar);
IC = zeros(nTypes*(nTypes-1),nvar);

% Build objective and constraints
for i = 1:nTypes
    f((i-1) * nSig + 1 : i*nSig) = -p(i)*sig(i,:,i);
    IR(i,(i-1) * nSig + 1 : i*nSig) = sig(i,:,i);
    IR(i,(i-1 + nTypes) * nSig + 1 : (i+nTypes)*nSig) = -v(i)*sig(i,:,i);
    for j = 1:(nTypes-1)
        IC((i-1)*(nTypes-1)+j , (i-1) * nSig + 1 : i*nSig) = sig(i,:,i);
        IC((i-1)*(nTypes-1)+j , (i-1 + nTypes) * nSig + 1 : (i+nTypes)*nSig) = -v(i)*sig(i,:,i);
        ICind = mod((i+j-1) * nSig + 1 : (i+j)*nSig,nSig*nTypes);
        ICind(ICind == 0) = nSig*nTypes;
        sigind = mod(i+j,nTypes);
        sigind(sigind == 0) = nTypes;
        IC((i-1)*(nTypes-1)+j,ICind) = -sig(sigind,:,i);
        IC((i-1)*(nTypes-1)+j,ICind + nTypes*nSig) = v(i)*sig(sigind,:,i);
    end
end

% Build linprog input
A = [IR;IC];
b = zeros(nTypes^2,1);
lb = zeros(nvar,1);
ub = ones(nvar,1);
lb(1:nSig*nTypes) = S;
ub(1:nSig*nTypes) = T;
opts = optimoptions('linprog','Display','off');
opts.ConstraintTolerance = 1e-6;
opts.OptimalityTolerance = 1e-8;

% Build problem structure
problem.f = f;
problem.Aineq = A;
problem.bineq = b;
problem.lb = lb;
problem.ub = ub;
problem.solver = 'linprog';
problem.options = opts;

% Execute linprog
[mech,rev,flag,output,LM] = linprog(problem);
rev = -rev;
end