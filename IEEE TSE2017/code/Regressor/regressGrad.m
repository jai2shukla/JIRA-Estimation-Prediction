function [ll,dl] = regressGrad(param,opt)

data		= opt.data;
output		= opt.output;
l1_penalty	= opt.l1_penalty;
l2_penalty	= opt.l2_penalty;
dataWeights	= opt.dataWeights;

dataSize = length(output);

predict = data*param';

err = predict - output;
ll	= -0.5*sum(dataWeights.*err.*err);

dl	= -(dataWeights.*err)'*data;

ll = ll / dataSize - 0.5*l2_penalty*param*param';
dl = dl ./ dataSize - l2_penalty*param;

epsilon = 1e-10;

% Pseudo-Huber style
norm1 = sqrt(epsilon + param.*param);
ll = ll  - l1_penalty*sum(norm1);
dl = dl  - l1_penalty*param ./ norm1;

% Pseudo-bound opt
%param2 = epsilon + abs(param);
%ll = ll  - l1_penalty*sum(0.5*param.*param ./ param2 + 0.5*param2);
%dl = dl  - l1_penalty*param ./ param2;


% -- manifold structure ----------
if isfield(opt,'ManifoldCorrel') & isfield(opt,'manifold_penalty') & ~isempty(opt.ManifoldCorrel) & opt.manifold_penalty

	data2 = data;
	
	if isfield(opt,'SideData')
		data2 = [data2;opt.SideData];
	end
	
	predict2 = data2*param';
	dataSize2 = size(data2,1);

	ll = ll  - 0.5*(opt.manifold_penalty*predict2'*opt.ManifoldCorrel*predict2) / dataSize2;
	dl = dl  - opt.manifold_penalty*((opt.ManifoldCorrel*predict2)'*data2) ./ dataSize2;
end
