function [sol,fvals,it,timex] = regress(data,output,param,opt)

% Training standard quadratic regression


%default values
nIters			= 1000;
epsilon			= 1e-7;
report_interval	= 10; %seconds
l1_penalty		= 0;
l2_penalty		= 1e-5;

if isfield(opt,'nIters')
	nIters = opt.nIters;
end

if isfield(opt,'epsilon')
	epsilon	= opt.epsilon;
end

if isfield(opt,'report_interval')
	report_interval	= opt.report_interval;
end

if isfield(opt,'l1_penalty')
	l1_penalty = opt.l1_penalty;
end

if isfield(opt,'l2_penalty')
	l2_penalty = opt.l2_penalty;
end


opt2 			= opt;
opt2.data		= data;
opt2.output		= output;
opt2.l1_penalty	= l1_penalty;
opt2.l2_penalty	= l2_penalty;

dataWeights		= ones(size(output));
if isfield(opt,'dataWeights')
	dataWeights = opt.dataWeights;
end
opt2.dataWeights = dataWeights;

% manifold learning
if isfield(opt2,'ManifoldStruct') & isfield(opt2,'manifold_penalty') & ~isempty(opt2.ManifoldStruct) & opt2.manifold_penalty

	ManifoldStyle = 'correlation';
	if isfield(opt2,'ManifoldStyle')
		ManifoldStyle = opt2.ManifoldStyle;
	end

	D = size(opt2.ManifoldStruct,1);

	ManifoldStruct = opt2.ManifoldStruct;

	%self-similarity is ignored
	ManifoldStruct(eye(D)==1) = 0; 

	if strcmpi(ManifoldStyle,'LLE')

		%Make sure that the *outlinks* are normalised
		ManifoldStruct = ManifoldStruct ./ (1e-5 + repmat(sum(ManifoldStruct,2),1,D));

		%Construct the correlation matrix
		tmp = sparse(eye(D)) - ManifoldStruct;
		opt2.ManifoldCorrel = tmp'*tmp;

	elseif strcmpi(ManifoldStyle,'Laplacian')

		%Construct the correlation matrix
		opt2.ManifoldCorrel = sparse(diag(sum(ManifoldStruct,2)))-ManifoldStruct;

	else % Default: purely association

		opt2.ManifoldCorrel = sparse(eye(D)) - ManifoldStruct;

	end
end

[dataSize,N] = size(data);

if isempty(param)
	param = zeros(1,N);
end

%[sol, fvals, it,timex] = conjugate_gradient('regressGrad',param,opt2,epsilon,nIters,report_interval);

[sol, fvals, it,timex] = LBFGS('regressGrad',param,opt2,10,0.01,epsilon,nIters,report_interval);

%second round
% featIds = find(abs(sol) >= 1e-4);
% opt2.data = opt2.data(:,featIds);
% [sol2, fvals, it,timex] = LBFGS('regressGrad',param(featIds),opt2,10,0.01,epsilon,nIters,report_interval);
% sol = zeros(size(param));
% sol(featIds) = sol2;
