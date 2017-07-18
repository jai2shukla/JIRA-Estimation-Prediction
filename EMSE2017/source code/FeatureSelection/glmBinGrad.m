function [ll,dl] = glmBinGrad(param,opt)

%
% supports
%	- reguarisation toward zeros: L1/L2-norm
%	- reguarisation toward prior params: L1/L2-norm
%	- regularisation toward parameter signs: L2-norm
% 	- regularisation toward prior structure: L2-norm
%

model		= opt.model;
data		= opt.data;
label		= opt.label;
dataSize 	= length(label);

%--- hyper-parameters ---
l1_penalty	= opt.l1_penalty;
l2_penalty	= opt.l2_penalty;

oscar_penalty = 0;
if isfield(opt,'oscar_penalty')
	oscar_penalty = opt.oscar_penalty;
end

biasId = length(param);
if isfield(opt,'biasId')
	biasId = opt.biasId;
end

nonBiasIds = setdiff([1:length(param)],biasId);

if isfield(opt,'l1_smooth')
	epsilon	= opt.l1_smooth;
else
	epsilon = 1e-5;
end

norm1_approx = 'huber';
if isfield(opt,'norm1_approx')
	norm1_approx = opt.norm1_approx;
end

param_signs = [];
if isfield(opt,'param_signs')
	param_signs = opt.param_signs;
end

sign_penalty = 0;
if isfield(opt,'sign_penalty')
	sign_penalty = opt.sign_penalty;
end

prior_param = [];
if isfield(opt,'prior_param')
	prior_param = opt.prior_param;
end

prior_l1_penalty = [];
if isfield(opt,'prior_l1_penalty')
	prior_l1_penalty = opt.prior_l1_penalty;
end

prior_l2_penalty = [];
if isfield(opt,'prior_l2_penalty')
	prior_l2_penalty = opt.prior_l2_penalty;
end

PriorCorrel = [];
if isfield(opt,'PriorCorrel')
	PriorCorrel 	= opt.PriorCorrel; %N2 x N2
end

prior_correl_penalty = 0;
if isfield(opt,'prior_correl_penalty')
	prior_correl_penalty = opt.prior_correl_penalty;
end

DataCorrel = [];
if isfield(opt,'DataCorrel')
	DataCorrel = opt.DataCorrel; %D x D
end

manifold_penalty = 0;
if isfield(opt,'manifold_penalty')
	manifold_penalty = opt.manifold_penalty;
end

isPositive	= 0;
if isfield(opt,'isPositive')
	isPositive = opt.isPositive;
end

weights	= ones(dataSize,1);
if isfield(opt,'weights')
	weights = opt.weights;
end

loss = 'log';
if isfield(opt,'loss')
	loss = opt.loss;
end

%--- end hyper-parameters ---


if isPositive
	param2 = param(nonBiasIds);
	param2(param2 < 0) = 0;
	param(nonBiasIds) = param2;
end

vals = data*param';

if strcmpi(loss,'hinge')

	y = label;
	y(y==0) = -1;
	lls = 1-y.*vals;
	
	dl = (weights.*y.*(lls > 0))'*data;
	
	ll = -sum(weights.*lls.*(lls > 0));
	
else %strcmpi(loss,'log')
	if strcmp(upper(model),'PROBIT') | strcmp(upper(model),'GAUSS') | strcmp(upper(model),'GAUSSIAN') | strcmp(upper(model),'NORMAL')

		probs   = 1 - cdf('norm',-vals,0,1);
		dl = (weights.*(label - probs).*  normpdf(-vals) ./ (1e-10 + probs .* (1-probs)))'*data;

	elseif strcmp(upper(model),'GUMBEL')

		probs   = 1 - exp(-exp(vals));
		dl = (weights.*(label - probs).* exp(vals) ./ (1e-10 + probs))'*data;

	elseif strcmp(upper(model),'WEIBULL') | strcmp(upper(model),'GOMPERTZ')

		probs   = exp(-exp(vals));
		dl = -(weights.*(label - probs).* exp(vals) ./ (1e-10 + 1-probs))'*data;

	else %LOGIT as default

		probs	= 1./ (1 + exp(-vals));
		dl = (weights.*(label - probs))'*data;

	end

	ll = (weights.*label)'*log(1e-10 + probs) + (weights.*(1-label))'*log(1e-10 + 1-probs);
end

% -- data manifold ----------
if ~isempty(DataCorrel)
	ll = ll  - 0.5*manifold_penalty*vals'*DataCorrel*vals;
	dl = dl  - manifold_penalty*(DataCorrel*vals)'*data;
end

ll = ll/sum(weights);
dl = dl./sum(weights);

param2 = param(nonBiasIds);

if isPositive
	dl2 = dl(nonBiasIds);
	dl2(dl2 < 0) = 0;
	dl(nonBiasIds) = dl2;
end

ll = ll - 0.5*l2_penalty*param2*param2';
dl(nonBiasIds) = dl(nonBiasIds) - l2_penalty*param2;

%--- l1_norm --------------
if l1_penalty

	if strcmp(norm1_approx,'huber')

		smallIds = find(abs(param2) < epsilon);
		largeIds = find(abs(param2) >= epsilon);

		norm1Grad = zeros(size(param2));
		norm1Gradx = zeros(size(param2));

		norm1 = sum(0.5*(param2(smallIds).^2)/epsilon) + sum(abs(param2(largeIds))-0.5*epsilon);
		norm1Grad(smallIds) = param2(smallIds)./epsilon;
		norm1Grad(largeIds) = sign(param2(largeIds));

	else %pseudo-huber

		norm1 = sum(sqrt(param2.^2 + epsilon*epsilon));
		norm1Grad = param2 ./ sqrt(param2.^2 + epsilon*epsilon);

	end

	ll = ll - l1_penalty*norm1;
	dl(nonBiasIds) = dl(nonBiasIds) - l1_penalty*norm1Grad;
end

%--- OSCAR-norm ---
if oscar_penalty %pseudo-huber
	N2 = length(param2);
	[sortVals,sortIds] = sort(abs(param2),'ascend');

	ranks = [1:N2];
	ranks(sortIds) = [1:N2];

	norm1s = sqrt(param2.^2 + epsilon*epsilon);
	norm1Grad = param2 ./ norm1s;

	ll = ll - oscar_penalty*sum((ranks-1).*norm1s)/N2;
	dl(nonBiasIds) = dl(nonBiasIds) - oscar_penalty*((ranks-1).*norm1Grad)./N2;
end

%--- parameter signs --------------
if ~isempty(param_signs)
	
	param3 = param2.*param_signs;
	
	ll = ll - 0.5*sign_penalty*sum(param2(param3 < 0).^2);	
	dl(nonBiasIds) = dl(nonBiasIds) - sign_penalty*param2(param3 < 0);
end

%--- prior parameters ------------
if ~isempty(prior_param)

	param3 = param - prior_param;
	

	ll = ll - 0.5*prior_l2_penalty*param*param';
	dl = dl - prior_l2_penalty*param;
	

	%--- l1_norm --------------
	if prior_l1_penalty
		if strcmp(norm1_approx,'huber')

			smallIds = find(abs(param3) < epsilon);
			largeIds = find(abs(param3) >= epsilon);

			norm1Grad = zeros(size(param3));
			norm1Gradx = zeros(size(param3));

			norm1 = sum(0.5*(param3(smallIds).^2)/epsilon) + sum(abs(param3(largeIds))-0.5*epsilon);
			norm1Grad(smallIds) = param3(smallIds)./epsilon;
			norm1Grad(largeIds) = sign(param3(largeIds));

		else %pseudo-huber

			norm1 = sum(sqrt(param3.^2 + epsilon*epsilon));
			norm1Grad = param3 ./ sqrt(param3.^2 + epsilon*epsilon);

		end

		ll = ll - prior_l1_penalty*norm1;
		dl = dl - prior_l1_penalty*norm1Grad;
	end
end

% -- feature correlation structure ----------
if ~isempty(PriorCorrel)
	ll = ll  - 0.5*prior_correl_penalty*param*PriorCorrel*param';
	dl = dl  - prior_correl_penalty*param*PriorCorrel';
end

