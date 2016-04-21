function [output] = glmBin(model,data,param,opt,phase)
%
% Generalized Linear Models for Binary outcomes
%
% Input:
%	data: 	each row is a data vector
%	label:  a column of 0/1 label
%	nIters: 	maximum number of iterations
%	l1_penalty: Laplacian penalty, often in the range [1e-5,1e-2];
%	l2_penalty: quadratic penalty, often in the range [1e-5,1e-2];
%	epsilon	:	threshold at which the learning stops if the relative improvement in log-likelihood falls below it.
%	report_interval: time interval to report the objective function (regularised likelihood). Set a very large number if you don't want to see report.
%
% Output:
%

[dataSize,N] = size(data);
start = tic();

if strcmpi(phase,'train') | strcmpi(phase,'training') | strcmpi(phase,'learn') | strcmpi(phase,'learning')

	%default values
	nIters			= 1000;
	epsilon			= 1e-7;
	report_interval	= 10; %seconds
	l1_penalty		= 0;
	l2_penalty		= 1e-5;
	isDropOut		= 0;
	isDropIn		= 0;
	
	if isfield(opt,'nIters')
		nIters = opt.nIters;
	end

	if isfield(opt,'nIterations')
		nIters = opt.nIterations;
	end
	
	if isfield(opt,'epsilon')
		epsilon	= opt.epsilon;
	end
	
	if isfield(opt,'report_interval')
		report_interval	= opt.report_interval;
	end

	if ~isfield(opt,'l1_penalty')
		opt.l1_penalty = l1_penalty;
	end

	if ~isfield(opt,'l2_penalty')
		opt.l2_penalty = l2_penalty;
	end

	if isfield(opt,'isDropOut')
		isDropOut = opt.isDropOut;
	end

	if isfield(opt,'isDropIn')
		isDropIn = opt.isDropIn;
	end

	opt.model 	= model;
	opt.data 	= data;
	
	if isempty(param)
		param = zeros(1,N);
	end
	
	if isfield(opt,'isAddedBias') & opt.isAddedBias
		param 	= [0,param];
		opt.data = [ones(dataSize,1),data];
	end

	N2 = length(param);

	%Ref: Sandler et al, "Regularized Learning with Networks of Features", NIPS'08 
	if isfield(opt,'PriorStruct') & isfield(opt,'prior_correl_penalty') & ~isempty(opt.PriorStruct) & opt.prior_correl_penalty
		
		PriorStyle = 'correlation';
		if isfield(opt,'PriorStyle')
			PriorStyle = opt.PriorStyle;
		end
		
		PriorStruct = opt.PriorStruct;

		%self-similarity is ignored
		PriorStruct(eye(N2)==1) = 0; 
		
		if strcmpi(PriorStyle,'LLE')
			
			%Make sure that the *outlinks* are normalised
			PriorStruct = PriorStruct ./ (1e-5 + repmat(sum(PriorStruct,2),1,N2));

			%Construct the correlation matrix
			%tmp = sparse(eye(N2)) - PriorStruct; %use this when stand alone nodes has "dummy" neighbors
			tmp = sparse(diag(sum(PriorStruct,2))) - PriorStruct; %standalone node is ignored
			
			opt.PriorCorrel = tmp'*tmp;
		elseif strcmpi(PriorStyle,'Laplacian')

			%Construct the correlation matrix
			opt.PriorCorrel = sparse(diag(sum(PriorStruct,2)))-PriorStruct;
			
		else % Default: purely association
		
			%opt.PriorCorrel = sparse(eye(N2)) - PriorStruct;
			opt.PriorCorrel =  - PriorStruct;
			
		end
	end

	if isfield(opt,'DataManifold') & isfield(opt,'manifold_penalty') & ~isempty(opt.DataManifold) & opt.manifold_penalty
		
		ManifoldStyle = 'Laplacian';
		if isfield(opt,'ManifoldStyle')
			ManifoldStyle = opt.ManifoldStyle;
		end
		
		DataManifold = opt.DataManifold;

		%self-similarity is ignored
		DataManifold(eye(dataSize)==1) = 0; 
		
		if strcmpi(ManifoldStyle,'LLE')
			
			%Make sure that the *outlinks* are normalised
			DataManifold = DataManifold ./ (1e-5 + repmat(sum(DataManifold,2),1,dataSize));

			%Construct the correlation matrix
			tmp = sparse(eye(dataSize)) - DataManifold;
			opt.DataCorrel = tmp'*tmp;
			
		else %if strcmpi(ManifoldStyle,'Laplacian') //default

			%Construct the correlation matrix
			opt.DataCorrel = sparse(diag(sum(DataManifold,2)))-DataManifold;

		end
	end
	
	if isDropOut | isDropIn
		[param] = dropout('glmBinGrad',param,opt,epsilon,nIters,report_interval);
	else
		param = conjugate_gradient('glmBinGrad',param,opt,epsilon,nIters,report_interval);
		%[param] = LBFGS('glmBinGrad',param,opt,10,0.01,epsilon,nIters,report_interval);
	end

	%--	
	output = param;

else %TESTING PHASE
	
	if length(param) == N + 1
		data = [ones(dataSize,1),data];
	end

	vals = data*param';

	if strcmpi(upper(model),'PROBIT') | strcmpi(upper(model),'GAUSS') | strcmpi(upper(model),'GAUSSIAN') | strcmpi(upper(model),'NORMAL')

		probs   = 1 - cdf('norm',-vals,0,1);

	elseif strcmpi(upper(model),'GUMBEL')
	
		probs   = 1 - exp(-exp(vals));
		
	elseif strcmpi(upper(model),'WEIBULL') | strcmpi(upper(model),'GOMPERTZ')
	
		probs   = exp(-exp(vals));

	else %strcmpi(upper(model),'LOGIT')

		probs	= 1./ (1 + exp(-vals));

	end

	output = probs;
end

