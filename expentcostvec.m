function [cost,grad] = expentcostvec(p,sig)
%EXPENTCOSTVEC Calculate expected entropy cost with gradient (vectorised)

[nTypes,nSig,~] = size(sig);
nVarsMech = 2*nSig*nTypes;

[temp_cost,temp_grad] = mutinfvec(p,permute(sig,[3,2,1]));
cost = p(:)' * temp_cost';

temp_grad = p(:) .* permute(temp_grad,[3,2,1]);
temp_grad = permute(temp_grad,[2,1,3]);
grad = [temp_grad(:);zeros(nVarsMech,1)];

end