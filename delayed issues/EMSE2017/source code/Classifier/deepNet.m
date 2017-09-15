function [output] = deepNet(data,param0,opt,outputMode,mode)

% Training | predicting with the deep neural network. Output is  softmax | logistic regression
% Fields:
%	param0
%		param0.Label
%		param0.LabelHidden
%		param0.Hidden{1...Depth}
%		param0.HiddenMap{1...Depth}
%
%	opt
%		opt.hiddenSizes(1...Depth)
%		opt.label
%		opt.epsilon
%		opt.nIters
%		opt.report_interval
%
%	output [mode = training]
%		output.Label
%		output.LabelHidden
%		output.Hidden{1...Depth}
%		output.HiddenMap{1...Depth}
%
%	output  [mode = testing]
%		output = labelProbs
%

[dataSize,N] = size(data);

start = tic();
last_time = toc(start);

hidDropOutRate = 0;
if isfield(opt,'hidDropOutRate')
	hidDropOutRate = opt.hidDropOutRate;
end

hidDropOutRates = [];
if isfield(opt,'hidDropOutRates')
	hidDropOutRates = opt.hidDropOutRates;
end

selectDropRate = 0;
if isfield(opt,'selectDropRate')
	selectDropRate = opt.selectDropRate;
end

visDropOutRate = 0;
if isfield(opt,'visDropOutRate')
	visDropOutRate = opt.visDropOutRate;
end

activationType = 'sigm';
if isfield(opt,'activationType')
	activationType = opt.activationType;
end

isGradAscent = 0;
if isfield(opt,'isGradAscent')
	isGradAscent = opt.isGradAscent;
end

transformRate = 0;
if isfield(opt,'transformRate')
	transformRate = opt.transformRate;
end

if ~(strcmpi(mode,'pretrain') | strcmpi(mode,'unsupervised') | strcmpi(mode,'hidden'))
	if ~isempty(param0)
		L =	length(param0.Label);
		opt.LabelSize 	= L;
	else
		L = opt.LabelSize;
	end
end

if strcmpi(mode,'train') | strcmpi(mode,'training') | strcmpi(mode,'learn') | strcmpi(mode,'learning') | strcmpi(mode,'pretrain')
	%-- hyper-parameter setting ----
	halfLifeLearnRate	= 10000; %iterations
	if isfield(opt,'halfLifeLearnRate')
		halfLifeLearnRate = opt.halfLifeLearnRate;
	end
	epsilon = 1e-5;
	if ~isfield(opt,'epsilon');
		opt.epsilon = epsilon;
	end
	
	if ~isfield(opt,'nIters')
		opt.nIters = 100;
	end
	
	if ~isfield(opt,'report_interval')
		opt.report_interval = 1;
	end

	if ~isfield(opt,'l1_penalty')
		opt.l1_penalty = 0;
	end
	
	if ~isfield(opt,'l2_penalty')
		opt.l2_penalty = 1e-4;
	end
	
	if ~isfield(opt,'norm_penalty')
		opt.norm_penalty = 0;
	end

	if ~isfield(opt,'MaxNorm')
		opt.MaxNorm = 100;
	end

	if ~isfield(opt,'Verbose')
		opt.Verbose = 0;
	end

	nConts = 0;
	if isfield(opt,'nConts')
		nConts = opt.nConts;
	end
	
	blockSize = 100;
	if isfield(opt,'blockSize')
		blockSize = opt.blockSize;
	end

	learnRate = 0.1;
	if isfield(opt,'learnRate')
		learnRate = opt.learnRate;
	end
	
	MaxNoImproves = 10;
	if isfield(opt,'MaxNoImproves')
		MaxNoImproves = opt.MaxNoImproves;
	end
	
	if strcmpi(outputMode,'softmax') & size(opt.label,2) == 1
		%represent the labels in the sparse matrix form
		label2 = zeros(dataSize,L);
		for d=1:dataSize
			label2(d,opt.label(d)) = 1;
		end
		opt.label		= label2;
	end
	%-- END hyper-parameter setting ----

	if ~isempty(param0) & ~isfield(opt,'hiddenSizes')
		for depth=1:length(param0.Hidden)
			opt.hiddenSizes(depth) = length(param0.Hidden{depth});
		end
	end

	Depth = length(opt.hiddenSizes);
	opt.Depth = Depth;

	%param initialization
	if isempty(param0)
		K = opt.hiddenSizes(1);
		for depth=1:Depth %top-down
			if depth < Depth
				K2 = opt.hiddenSizes(depth+1);
			else
				K2 = N;
			end

			param0.Hidden{depth} 	= zeros(1,K);
			param0.HiddenMap{depth}	= 0.1*randn(K,K2)/sqrt(K2);
	
			K = K2;
		end
		
		L = opt.LabelSize;
		K = opt.hiddenSizes(1);
		param0.Label		= zeros(1,L);
		param0.LabelHidden	= 0.1*randn(L,K)/sqrt(K);
	end
	
	if ~isfield(opt,'activationTypes')
		for depth=1:Depth
			activationTypes{depth} = activationType;
		end
	else
		activationTypes = opt.activationTypes;
	end
end

if strcmpi(mode,'train') | strcmpi(mode,'training') | strcmpi(mode,'learn') | strcmpi(mode,'learning') | strcmpi(mode,'supervised')

	%--- SUPERVISED LEARNING ----------------
	L = opt.LabelSize;
	%--- reshape parameters--- 
	K = opt.hiddenSizes(1);
	param = [param0.Label,reshape(param0.LabelHidden,1,L*K)];
	for depth=1:Depth
		if depth < Depth
			K2 = opt.hiddenSizes(depth+1);
		else
			K2 = N;
		end
		param = [param,param0.Hidden{depth},reshape(param0.HiddenMap{depth},1,K*K2)];

		K = K2;
	end
	%--- END reshape parameters ----

	opt2 = opt;
	opt2.outputMode = outputMode;

	
	preLL = -1e+5;

	randIds = randperm(dataSize);

	learnRate0 = learnRate;
	iter2 = 0;
	noImproves = 0;

	for iter=1:opt.nIters
		iter2 = iter2 + 1;

		p = 0;
		LL = 0;
		for d=1:blockSize:dataSize
			p = p + 1;
			dMax = min(d+blockSize-1,dataSize);
			thisBlockSize = dMax-d+1;

			opt2.data	= data(randIds(d:dMax),:);
			opt2.label	= opt.label(randIds(d:dMax),:);

			if transformRate & rand < 0.5
				norm1	= norm(opt2.data(:));
				tmp 	= opt2.data*randn(N);
				opt2.data = (1-transformRate)*opt2.data + transformRate*tmp*norm1./norm(tmp(:));
			end

			if visDropOutRate
				opt2.data(rand(size(opt2.data)) < visDropOutRate)= 0;
			end

			if hidDropOutRate | ~isempty(hidDropOutRates)
				for depth=1:Depth
					K = opt.hiddenSizes(depth);
					opt2.dropOut{depth} = false(thisBlockSize,K);

					if isempty(hidDropOutRates)
						opt2.dropOut{depth}	= (rand(thisBlockSize,K) < hidDropOutRate);
					else
						opt2.dropOut{depth}	= (rand(dMax-d+1,opt.hiddenSizes(depth)) < hidDropOutRates(depth));
					end
				end
			end

			[ll,dl] = deepNetGrad(param,opt2);
			
			%for robustness
			%dl(dl > 10) 	= 10;
			%dl(dl < -10)	= -10;

			LL = LL + ll;
			param = param + learnRate*dl;
		end

		if iter2 >= halfLifeLearnRate | (iter > 1 & (preLL-LL)/abs(preLL) > 3e-2) | noImproves >= MaxNoImproves
			learnRate = 0.5*learnRate;
			if isfield(opt,'gradNorm')
				opt.gradNorm = 0.5*opt.gradNorm;
			end

			iter2 = 0;
			noImproves = 0;

			if learnRate < 0.001
				break;
			end

			fprintf('Adjust learning rate: %.3f\n',learnRate);
		end

		if (preLL-LL)/abs(preLL) > 0
			noImproves = noImproves + 1;
		else
			%noImproves = 0;
		end

		if abs((LL - preLL)/preLL) < 1e-5
			%break;
		end
		preLL = LL;

		curr_time = toc(start);
		if curr_time >= last_time + opt.report_interval
			fprintf('Iter: %d, batch: %d, ll: %.5f, time: %.1f\n',iter,p,LL/p,curr_time);
			last_time = curr_time;
		end
	end

	%--- reshape parameters ------
	output.Label		= param(1:L); lastF = L;
	K = opt.hiddenSizes(1);
	output.LabelHidden	= (1-hidDropOutRate)*reshape(param(lastF+1:lastF+L*K),L,K); lastF = lastF+L*K;

	for depth=1:Depth

		output.Hidden{depth}	= param(lastF+1:lastF+K); lastF = lastF+K;
		
		if depth < Depth
			K2 = opt.hiddenSizes(depth+1);
			output.HiddenMap{depth}	= (1-hidDropOutRate)*reshape(param(lastF+1:lastF+K*K2),K,K2); lastF = lastF+K*K2;
		else
			K2 = N;
			output.HiddenMap{depth}	= (1-visDropOutRate)*reshape(param(lastF+1:lastF+K*K2),K,K2); lastF = lastF+K*K2;
		end

		K = K2;
	end
	
	%----- END reshape parameters ---------
elseif strcmpi(mode,'test') | strcmpi(mode,'testing') | strcmpi(mode,'predict') | strcmpi(mode,'predicting') | ...
		strcmpi(mode,'prediction')

	Depth = length(param0.Hidden);
	
	if ~isfield(opt,'activationTypes')
		for depth=1:Depth
			activationTypes{depth} = activationType;
		end
	else
		activationTypes = opt.activationTypes;
	end

	hiddenVals = ones(dataSize,1)*param0.Hidden{Depth} +  data*param0.HiddenMap{Depth}';
	hiddenActivations{Depth} = getActivation(hiddenVals,activationTypes{Depth});

	if selectDropRate
		hiddenActivations{Depth}(hiddenActivations{Depth} < prctile(hiddenActivations{Depth}(:),selectDropRate*100)) = 0;
	end
	
	for depth=Depth-1:-1:1
		hiddenVals = ones(dataSize,1)*param0.Hidden{depth} + hiddenActivations{depth+1}*param0.HiddenMap{depth}';
		hiddenActivations{depth} =  getActivation(hiddenVals,activationTypes{depth});

		if selectDropRate
			hiddenActivations{depth}(hiddenActivations{depth} < prctile(hiddenActivations{depth}(:),selectDropRate*100)) = 0;
		end
	end

	labelVals = ones(dataSize,1)*param0.Label + hiddenActivations{1}*param0.LabelHidden';
	
	
	L = length(param0.Label);
	if strcmpi(outputMode,'softmax')
		
		labelProbs	= exp(labelVals);
		Zs			= sum(labelProbs,2);
		labelProbs	= labelProbs ./ (Zs * ones(1,L));

		output = labelProbs;
	elseif strcmpi(outputMode,'logit')
	
		output = 1./ (1 + exp(-labelVals));
		
	elseif strcmpi(outputMode,'regress')
	
		output = labelVals;
	end
else
	fprintf('Err: Dont know the mode %s!!\n',mode);
end
