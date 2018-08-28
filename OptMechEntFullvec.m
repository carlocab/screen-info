function [ value , revenue , mechanism , sig , exitflag , output , LM , ...
    grad , hessian ] = OptMechEntFullvec( p , v , T , S , g , nSig , varargin )
%OPTMECHENTFULL Optimal screening with information acquisition
%   Detailed explanation goes here

% Check if inputs are valid
if length(p) ~= length(v) || min([size(p),size(v)]) > 1
    error('p and v must be vectors of the same length!');
elseif ~ismembertol(sum(p),1) || ~isempty(p(p < 0))
    warning('p is not a probability distribution.');
end

nTypes = length(p);
if nSig > 1 % Run non-linear solver only if nSig > 1
    % Identify number of optimization variables
    nVarsMech = 2*nSig*nTypes;
    nVarsSig = nSig*nTypes^2;
    nVars = nVarsMech + nVarsSig;

    if ~isempty(varargin) && strcmp(varargin{1},'random')
        % Initialise random x0
        x0 = rand(nVars,1);
        for i = 1:nTypes^2
            x0((i-1)*nSig+1:i*nSig) = x0((i-1)*nSig+1:i*nSig) / sum(x0((i-1)*nSig+1:i*nSig));
        end
    else
        % Initialise x0 to be very informative
        x0 = zeros(nVars,1);
        typeind = 1:nTypes;
        sigind = mod(typeind,nSig);
        sigind(sigind == 0) = nSig;
        for i = typeind-1
            x0(sub2ind([nSig,nTypes,nTypes],sigind,circshift(typeind,i),typeind))...
                = 1;
        end
    end

    % Initalise x0 with optimal trading mechanism given signals
    [~,x0(nVarsSig+1:end)] = revinfoLP(p,v,T,S,...
        permute(reshape(x0(1:nVarsSig),nSig,nTypes,nTypes),[2,1,3]));

    % Construct probability constraints
    A = zeros(nTypes^2,nVars);
    colind = 1:nVarsSig;
    colind = reshape(colind,[nSig,nTypes^2])';
    Aind = sub2ind([nTypes^2,nVars],repmat((1:(nTypes^2))',1,nSig),colind);
    A(Aind) = 1;

    b = ones(nTypes^2,1);

    % Construct upper and lowerbounds
    lb = zeros(nVars,1);
    ub = ones(nVars,1);
    if S ~= 0
        lb(nVarsSig + 1 : nVarsSig + nSig*nTypes) = S;
    end
    ub(nVarsSig + 1 : nVarsSig + nSig*nTypes) = T;

    % Construct objective function to minimise
    obj = @(x) fullobjvec(x,p,g,nSig);

    % Construct non-linear constraint
    nlcon = @(x) mechconstrvec(v,x(nVarsSig+1:nVars),...
        permute(reshape(x(1:nVarsSig),nSig,nTypes,nTypes),[2,1,3]));

    % fmincon options
    opts = optimoptions('fmincon');
    opts.Display = 'notify-detailed';
    opts.MaxFunctionEvaluations = 1e8;
    opts.MaxIterations = 1e4;
    opts.SpecifyObjectiveGradient = true;
    opts.SpecifyConstraintGradient = true;
    opts.FiniteDifferenceType = 'central';
    opts.FunValCheck = 'on';

    % Create problem instance
    problem = struct;
    problem.solver = 'fmincon';
    problem.x0 = x0;
    problem.objective = obj;
    problem.lb = lb;
    problem.ub = ub;
    problem.Aeq = A;
    problem.beq = b;
    problem.nonlcon = nlcon;
    problem.options = opts;

    % Run fmincon
    [x,value,exitflag,output,LM,grad,hessian] = fmincon(problem);

    % Check if hessian is negative definite
    [~,pd] = chol(hessian);
    if pd
        warning('OPT:SOC','Hessian matrix is not negative definite.');
    end

    % Construct output
    value = -value;
    grad = -grad;
    hessian = -hessian;
    sig = permute(reshape(x(1:nVarsSig),nSig,nTypes,nTypes),[2,1,3]);
    mechanism = x(nVarsSig+1:nVars);
    revenue = exprevvec(p,x(nVarsSig+1:nVars),permute(reshape(x(1:nVarsSig),...
        nSig,nTypes,nTypes),[2,1,3]));

else % Solve linear programming problem directly when nSig <= 1
    sig = ones(nTypes,1,nTypes);
    [revenue,mechanism,LM,exitflag,output] = revinfoLP(p,v,T,S,sig);
    value = revenue;
    [grad,hessian] = deal([]);
end

end
