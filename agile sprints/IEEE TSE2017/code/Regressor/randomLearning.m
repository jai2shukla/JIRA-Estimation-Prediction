function [output] = randomLearning(data,param,opt,phase)

% For multiple classes, assume that category 1 is the base class

%
%	* Combining many ideas from functional gradient boosting and random forests
%		- There are several restarts (or bootstraps)
%		- Each restart, the model is fitted sequentially: A new weak learner is added to the ensemble if it helps improving the evaluation metric
%	* Allowing custom (non-smooth) evaluation metrics
%	* Allowing custom (smooth) gradient
%		
%	* Training options
%		- opt.LearnerSet		: Set of possible weak learners.
%					opt.LearnerSet.Names
%									Possible subset of the following:
%									-> 'NNet' -- neural networks (Default)
%									-> 'PLS' -- partial least square (built-in)
%									-> 'LeastSquare' | 'RidgeRegress' | 'Lasso' | 'ElasticNet' | 'Linear'
%									-> 'RegressTree' -- regression tree (built-in)
%					opt.LearnerSet.Options
%									Options for each learner type
%

%		- opt.Metric			: Name of the evaluation metric. Default is RMSE for regression, and F1 for classification
%									Possible values: 'R2' | 'RMSE' | 'F1' | 'Accuracy'

%		- opt.Surrogate			: Name of the smooth surrogate function where functional gradient can be computed (e.g., can be the same as opt.Metric).
%									Possible values: 
%									-> 'R2' | 'RMSE' (Default)  for regression
%									-> 'Logit' (Default) | 'GumbelRegress' (for binary classification)
									
%		- opt.Dim				: Number of features picked each time. Default is sqrt(dataSize).
%		- opt.MaxRestart		: Maximum number of restartings. Default is 1.
%		- opt.MaxIter			: Maximum number of iterations for each restart. Default is 100.
%		- opt.MaxStepSize		: Maximum step size for after gradient fitting. Default is 0.1.
%		- opt.ReportInterval	: Reporting interval of the learning progress. Default is 1s.
%		- opt.SubsampleRate		: Data rate at which the weak learner is sampling. Default is 0.5.
%		- opt.MaxUnsuccessTrial	: Maximum number of unsuccessful trials on weak learners. Default is 50.

%		- opt.ValidData			: Data for validation (e.g., best number iterations)
%		- opt.ValidOutcome		: The label for validation
%		- opt.MinValidScore		: Mininum portion of validation performance compared to the best one. Default is 0.9

%	* Prediction options
%		- opt is the structure learned from the training phase
%

[dataSize,N] = size(data);

start = tic();
last_time = toc(start);

probType = 'regress';
if isfield(opt,'probType')
	probType = opt.probType;
end
if strcmpi(probType,'regress-mean') 
	probType = 'regress';
end

if strcmpi(phase,'train') %in the training phase

	%make the output binary matrix
	if isfield(opt,'LabelSize')
		L = opt.LabelSize;
	else
		L = 1;
	end

	if strcmpi(probType,'multiClassify') | strcmpi(probType,'softmax') | strcmpi(probType,'maxent')
		if size(opt.Outcome,2) == 1

			L = max(opt.Outcome);

			label = zeros(dataSize,L);
			for d=1:dataSize
				label(d,opt.Outcome(d)) = 1;
			end

			opt.Outcome = label;
		else
			L = size(opt.Outcome,2);
		end
	end
	
	%--- weak learner's name --
	LearnerName = 'NNet';
	LearnerSetSize = 1;
	
	%-- evaluation metric --
	Metric = 'Likelihood';
	
	if strcmpi(probType,'regress') 
		Metric = 'RMSE';
	elseif strcmpi(probType,'regress-median')
		Metric = 'MAE';
	end
	
	if isfield(opt,'Metric')
		Metric = opt.Metric;
	end
	
	%-- surrogate function --
	Surrogate = 'RMSE';
	
	if strcmpi(probType,'regress')
		Surrogate = 'RMSE';
	elseif strcmpi(probType,'regress-median')
		Surrogate = 'MAE';
	elseif strcmpi(probType,'multiClassify')
		Surrogate = 'Maxent';
	elseif strcmpi(probType,'binClassify')
		Surrogate = 'Logit';
	elseif strcmpi(probType,'PoissonRegress')
		Surrogate = 'PoissonRegress';
	elseif strcmpi(probType,'CumulOrd') | strcmpi(probType,'CumOrd')
		Surrogate = 'CumulOrd';
	end
	
	if isfield(opt,'Surrogate')
		Surrogate = opt.Surrogate;
	end
	
	
	%-- number of runs --
	MaxRestart = 1;
	if isfield(opt,'MaxRestart')
		MaxRestart = opt.MaxRestart;
	end

	%-- number of weak learners --
	MaxIter = 100;
	if isfield(opt,'MaxIter')
		MaxIter = opt.MaxIter;
	end

	%-- maximum step size --
	MaxStepSize	= 0.1;
	if isfield(opt,'MaxStepSize')
		MaxStepSize = opt.MaxStepSize;
	end
	stepSize = 0.1*MaxStepSize;
	
	%-- reporting interval --
	ReportInterval = 1;
	if isfield(opt,'ReportInterval')
		ReportInterval = opt.ReportInterval;
	end
	
	%-- unsuccessful trials -
	MaxUnsuccessTrial = 100;
	if isfield(opt,'MaxUnsuccessTrial')
		MaxUnsuccessTrial = opt.MaxUnsuccessTrial;
	end
	
	%-- min valid reduction -
	MinValidScore = 0.6;
	if isfield(opt,'MinValidScore')
		MinValidScore = opt.MinValidScore;
	end
	
	%-- subsample rate --
	SubsampleRate = 0.5;
	if isfield(opt,'SubsampleRate')
		SubsampleRate = opt.SubsampleRate;
	end
	
	if isfield(opt,'LearnerSet')
		LearnerSetSize = length(opt.LearnerSet.Names);

		for learnerId = 1:LearnerSetSize
			learnerSuccess{learnerId} = 0;
		end
	end
	

	Dim = zeros(1,LearnerSetSize) + min(N,floor(sqrt(dataSize)));
	if isfield(opt,'Dim')
		Dim = opt.Dim;
	end
	
	funcOpt = [];
	perfOpt = [];
	thresholds = [];
	if strcmpi(Surrogate,'CumulOrd')
		%L = opt.MaxLevel;
		L = max(opt.Outcome);
		funcOpt.MaxLevel = L;
		perfOpt.MaxLevel = L;
	end
	
	%--- THE MAIN LEARNING LOOP ---
	for restart=1:MaxRestart

		if strcmpi(Surrogate,'CumulOrd')
			thresholds = zeros(1,L-1);
			for l=2:L-1
				thresholds(l) = l;
			end
			perfOpt.Threshold = thresholds;
		end

		%-- re-starting --
		if MaxRestart > 1
			dataIds = randi(dataSize,[1,dataSize]);
		else
			dataIds = [1:dataSize];
		end
		
		refOutcomes		= opt.Outcome(dataIds,:);
		restartData		= data(dataIds,:);
		
		if strcmpi(Surrogate,'CumulOrd')
			pre_funcs	= zeros(dataSize,1);
		elseif(L > 1)
			pre_funcs	= zeros(dataSize,L);
		else
			pre_funcs	= zeros(dataSize,1);
		end
		
		preScore	= calcPerfScore(pre_funcs,refOutcomes,Metric,probType,perfOpt);

		if isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
			validSize = length(opt.ValidOutcome);
			bestValidScore	= calcPerfScore(zeros(validSize,1),opt.ValidOutcome,Metric,probType,perfOpt);
			validFuncs	= zeros(validSize,1);
		end
		

		iter2 = 0;
		bestIter = 0;
		unsuccessTrial = 0;
		for iter=1:MaxIter
		
			if strcmpi(Surrogate,'CumulOrd')
				funcOpt.Threshold = thresholds;
				perfOpt.Threshold = thresholds;
			end
			
			if isfield(opt,'LearnerSet')
				learnerId = ceil(rand*LearnerSetSize);
				LearnerName = opt.LearnerSet.Names{learnerId};
			end

			M = Dim(learnerId);			
		
			%--- fit the weak learner to the gradient of the surrogate function ----
			featIds = randperm(N); featIds = featIds(1:M);
			features = restartData(:,featIds);
			
			if isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
				validFeatures = opt.ValidData(:,featIds);
			end			
			
			% use a random subsample of the data for fitting
			trainSize = ceil(SubsampleRate*dataSize);	trainIds = randperm(dataSize); trainIds = trainIds(1:trainSize);

			if strcmpi(Surrogate,'CumulOrd') %move the threshold
				[trainFuncGrad,thresholdGrad] = getFuncGrad(pre_funcs(trainIds,:),refOutcomes(trainIds,:),Surrogate,funcOpt);
			else
				[trainFuncGrad] = getFuncGrad(pre_funcs(trainIds,:),refOutcomes(trainIds,:),Surrogate,funcOpt);
			end
			
			if isfield(opt.LearnerSet,'Opts')
				weakOpt = opt.LearnerSet.Opts{learnerId};
			end
			weakOpt.LabelSize = 1;
			weakLearner = fitFuncGrad(features(trainIds,:),trainFuncGrad,LearnerName,weakOpt);

			%--- search for the best step-size w.r.t. the metrics -----
			searchDirection = getSearchDirection(features,weakLearner,LearnerName,weakOpt);
			bestScore = calcPerfScore(pre_funcs,refOutcomes,Metric,probType,perfOpt);
			currStepSize = stepSize; bestStepSize = stepSize;

			isSucceeded = false;
			while bestStepSize < MaxStepSize

				funcs = pre_funcs + currStepSize*searchDirection;
				
				if strcmpi(Surrogate,'CumulOrd') %move the threshold
					perfOpt.Threshold = thresholds + currStepSize*tanh(thresholdGrad);

					for l=2:L-1  %forcing the monotonicity
						if perfOpt.Threshold(l) <= perfOpt.Threshold(l-1) + 0.5
							perfOpt.Threshold(l) = perfOpt.Threshold(l-1) + 0.5;
						end
					end
				end
				
				currScore = calcPerfScore(funcs,refOutcomes,Metric,probType,perfOpt);
				
				if currScore >= bestScore
					bestScore		= currScore;
					bestStepSize	= currStepSize;

					currStepSize =	1.1*currStepSize;
					isSucceeded = true;
				else
					break;
				end
			end

			if isSucceeded % pick this learner if there has been some improvement
				iter2 = iter2 + 1;

				delta_funcs = bestStepSize*searchDirection;
				
				pre_funcs 	= pre_funcs + delta_funcs;
				preScore	= calcPerfScore(pre_funcs,refOutcomes,Metric,probType,perfOpt);

				output{restart}{iter2}.WeakLearner.Name		= LearnerName;
				output{restart}{iter2}.WeakLearner.Struct	= weakLearner;
				output{restart}{iter2}.WeakLearner.Opts		= weakOpt;
				
				output{restart}{iter2}.FeatId 				= featIds;
				output{restart}{iter2}.StepSize 			= bestStepSize;
				output{restart}{iter2}.DeltaFunc 			= mean(delta_funcs);
				
				if strcmpi(Surrogate,'CumulOrd') %move the threshold
					thresholds = thresholds + bestStepSize*thresholdGrad;

					funcOpt.Threshold = thresholds;
					perfOpt.Threshold = thresholds;
				end
				
				if isfield(opt,'LearnerSet')
					learnerSuccess{learnerId} = learnerSuccess{learnerId} + 1;
				end
				
				unsuccessTrial = 0;
			else
				unsuccessTrial = unsuccessTrial + 1;
			end

			if unsuccessTrial > MaxUnsuccessTrial
				%break;
			end
			
			if isSucceeded & isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
				
				validFuncs = validFuncs + bestStepSize*getSearchDirection(validFeatures,weakLearner,LearnerName,weakOpt);
				validScore = calcPerfScore(validFuncs,opt.ValidOutcome,Metric,probType,perfOpt);
				
				if validScore > bestValidScore
					bestValidScore = validScore;
					bestIter = iter2;
				elseif validScore <= MinValidScore*bestValidScore %stop if the current learning leads to serious overfitting
					%break;
				end
			end

			curr_time = toc(start);
			if curr_time > last_time + ReportInterval
			
				if isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
					fprintf('\tRestart %d, iter: %d, learner: %d, train score: %.5f, valid score: %.5f, time: %.1f\n',restart,iter,iter2,full(preScore),full(validScore),curr_time);
				else
					fprintf('\tRestart %d, iter: %d, learner: %d, train score: %.5f, time: %.1f\n',restart,iter,iter2,full(preScore),curr_time);
				end
				
				last_time = curr_time;
			end
		end

		%stop the weak learner at the peak of the validation
		if isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
		
			bestValidScores(restart) = bestValidScore;
			
			clear bestOutput;
			
			for iter2=1:bestIter
				bestOutput{iter2} = output{restart}{iter2};
			end
			
			if bestIter
				output{restart} =  bestOutput;
			end
		end

		if strcmpi(probType,'CumulOrd') | strcmpi(probType,'SeqOrd')
			output{restart}{1}.Threshold = thresholds;
		end
	end
	
	%keep only those high performing restarts
	if isfield(opt,'ValidData') & isfield(opt,'ValidOutcome')
		[sortVals,sortIds] = sort(bestValidScores,'descend');
		
		for b2=1:ceil(MaxRestart/2)
			output2{b2} = output{sortIds(b2)};
		end
		
		output = output2;
	end

	if isfield(opt,'LearnerSet')
		fprintf('\tLearner success rate:\n');
		for learnerId=1:LearnerSetSize
			fprintf('\t\t%s : \t%d\n',opt.LearnerSet.Names{learnerId},learnerSuccess{learnerId});		
		end
	end
	
elseif strcmpi(phase,'test') %in the testing phase

	
	Restart = length(param);

	if strcmpi(probType,'CumulOrd') | strcmpi(probType,'SeqOrd')
		output = zeros(dataSize,opt.MaxLevel);
	end
	
	L = 1;
	if isfield(opt,'LabelSize')
		L = opt.LabelSize;
	end
	output = zeros(dataSize,L);
	
	for restart=1:Restart
		
		%get the functional vals
		funcVals = zeros(dataSize,L);
		for iter=1:length(param{restart})
			subdata 	= data(:,param{restart}{iter}.FeatId);
			weakLearner	= param{restart}{iter}.WeakLearner.Struct;
			LearnerName	= param{restart}{iter}.WeakLearner.Name;
			weakOpt 	= param{restart}{iter}.WeakLearner.Opts;

			funcVals = funcVals + param{restart}{iter}.StepSize*getSearchDirection(subdata,weakLearner,LearnerName,weakOpt);
		end

		if strcmpi(probType,'CumulOrd') | strcmpi(probType,'SeqOrd')
			
			Surrogate = probType;
			
			testOpt.Threshold = param{restart}{1}.Threshold;
			output = output + getProbs(funcVals,Surrogate,opt);

		elseif strcmpi(probType,'multiClassify')
			
			Surrogate = 'Maxent';
			output = output + getProbs(funcVals,Surrogate,opt);

		elseif strcmpi(probType,'binClassify')
		
			output = output + 1 ./ (1+exp(-funcVals));
			
		elseif strcmpi(probType,'PoissonRegress')
		
			output = output + exp(funcVals);
			
		else
		
			output = output + funcVals;
		end
		
	end
	
	output = output ./ Restart;
	
elseif strcmpi(phase,'feat') %compute the feature importance

	Restart = length(param);

	featScores = zeros(1,N);
	for restart=1:Restart
		for iter=1:length(param{restart})
			featScores(param{restart}{iter}.FeatId) = featScores(param{restart}{iter}.FeatId) + param{restart}{iter}.StepSize;
		end
	end	

	output = featScores/Restart;

elseif strcmpi(phase,'extract') %using weak learner as feature extractor

	Restart = length(param);


	N = 0;
	for restart=1:Restart
		N = N + length(param{restart});
	end

	features = zeros(dataSize,N);
	
	i = 0;
	for restart=1:Restart
		for iter=1:length(param{restart})
			i = i + 1;
			
			subdata 	= data(:,param{restart}{iter}.FeatId);
			weakLearner	= param{restart}{iter}.WeakLearner.Struct;
			LearnerName	= param{restart}{iter}.WeakLearner.Name;
			weakOpt 	= param{restart}{iter}.WeakLearner.Opts;
			features(:,i) = getSearchDirection(subdata,weakLearner,LearnerName,weakOpt);
		end
	end
	
	output = features;
	
else
	
	fprintf('ERR: unknown phase\n of the randomLearning()');
	
end

%---
function [score] = calcPerfScore(funcVals,trueOutcomes,Metric,probType,opt)

	if strcmpi(Metric,'R2')
	
		score = calcR2(funcVals,trueOutcomes);
		
	elseif strcmpi(Metric,'RMSE')

		score = -sqrt(sum((trueOutcomes-funcVals).^2)/length(trueOutcomes));
		
	elseif strcmpi(Metric,'MAE')
	
		score = -sum(abs(trueOutcomes-funcVals)/length(trueOutcomes));

	elseif strcmpi(probType,'binClassify')
	
		if strcmpi(Metric,'Accuracy')
		
			score = (sum(funcVals > 0 & trueOutcomes > 0) + sum(funcVals <= 0 & trueOutcomes <= 0)) / length(trueOutcomes);

		elseif strcmpi(Metric,'Prec') | strcmpi(Metric,'Precision')
		
			score = (sum(funcVals > 0 & trueOutcomes > 0)) / (1e-5 + sum(funcVals > 0));
					
		elseif strcmpi(Metric,'SensioSpecific')
		
			sensitivity		= sum(funcVals > 0 & trueOutcomes > 0) / (1e-10 + sum(trueOutcomes > 0));
			specificity 	= sum(funcVals <= 0 & trueOutcomes <= 0) / (1e-10 + sum(trueOutcomes <= 0));

			score = 0.5*(sensitivity+specificity);
			%score = 2*sensitivity*specificity/(1e-5 + sensitivity+specificity);
			%score = 0.5*((sensitivity+specificity) - abs(sensitivity-specificity));

		elseif strcmpi(Metric,'Specificity')
		
			score 	= sum(trueOutcomes < 0 & trueOutcomes <= 0) / (1e-10 + sum(trueOutcomes <= 0));

		elseif strcmpi(Metric,'AUC')
			
			score	= auc([trueOutcomes,funcVals],0.05,'mann-whitney');

		else %Default: (strcmpi(Metric,'Likelihood') | strcmpi(Metric,'CrossEntropy'))

			probs = 1 ./ (1+exp(-funcVals));		
			score = sum(log([probs(trueOutcomes==1);1-probs(trueOutcomes==0)]))/length(trueOutcomes);
		end
	
	elseif strcmpi(probType,'PoissonRegress')

		%Detault: strcmpi(Metric,'Likelihood')
		score = sum(trueOutcomes .* funcVals - exp(funcVals))/length(trueOutcomes);

	elseif strcmpi(probType,'CumulOrd') | strcmpi(probType,'CumOrd')


		dataSize = length(trueOutcomes);
		
		L = opt.MaxLevel;
		Surrogate = probType;
		
		[probs] = getProbs(funcVals,Surrogate,opt);

		if strcmpi(Metric,'Likelihood')
		
			score = 0;
			for l=1:L
				score = score + sum(log(1e-10 + probs(trueOutcomes==l,l)));
			end
			score = score/length(trueOutcomes);
			
		elseif strcmpi(Metric,'MacroF1')

			[maxVals,maxStates] = max(probs');
			maxStates = maxStates';
			
			macroF1 = 0;
			%for l=1:L
			for l=2:L
				recall	= sum(maxStates==l & trueOutcomes==l)/(1e-5 + sum(trueOutcomes==l));
				prec	= sum(maxStates==l & trueOutcomes==l)/(1e-5 + sum(maxStates==l));
				
				macroF1 = macroF1 + 2*prec*recall/(1e-5 + prec + recall);
			end
			macroF1 = macroF1/(L-1);
			%macroF1 = macroF1/L;
			
			score = macroF1;
		else
			fprintf('ERR: unknown metric %s\n',Metric);
			return;
		end

	elseif strcmpi(probType,'multiClassify')


		[dataSize,L] = size(trueOutcomes);
		Surrogate = 'Maxent';
	
		[probs] = getProbs(funcVals,Surrogate,opt);

		if strcmpi(Metric,'Likelihood')
			score = sum(log(1e-10 + probs(trueOutcomes==1)))/dataSize;
		else
			fprintf('ERR: unknown metric %s\n',Metric);
			return;
		end
		
	elseif strcmpi(Metric,'Top10') | strcmpi(Metric,'Top20') | strcmpi(Metric,'Top50') | strcmpi(Metric,'Top100')  | strcmpi(Metric,'Top200') | strcmpi(Metric,'Top500')
	
		topK = 100;
		if strcmpi(Metric,'Top10')
			topK = 10;
		elseif strcmpi(Metric,'Top20')
			topK = 20;
		elseif strcmpi(Metric,'Top50')
			topK = 50;
		elseif strcmpi(Metric,'Top100')
			topK = 100;
		elseif strcmpi(Metric,'Top200')
			topK = 200;
		elseif strcmpi(Metric,'Top500')
			topK = 500;
		end
	
		[sortVals,sortIds] = sort(funcVals,'descend');
		score = sum(trueOutcomes(sortIds(1:topK)) == max(trueOutcomes));
	else
		
		fprintf('ERR: unknown evaluation metric: %s\n',Metric);
		
	end

%- fitting weak learner to the functional gradients --
function [weakLearner] = fitFuncGrad(features,funcGrad,LearnerName,opt)

	[trainSize,M] = size(features);

	
	L = size(funcGrad,2);
	
	if strcmpi(LearnerName,'NNet')
	
		K = 5;
		if isfield(opt,'HiddenSize')
			K = opt.HiddenSize;
		end

		if isfield(opt,'activationType')
			opNNet.activationType = opt.activationType;
		end

		opNNet.nIters = 300;
		if isfield(opt,'nIters')
			opNNet.nIters = opt.nIters;
		end

		
		param0.Label 			= zeros(1,L);
		param0.LabelHidden 		= zeros(L,K);
		param0.Hidden			= zeros(1,K);
		param0.HiddenVisible	= 0.1*randn(K,M)/sqrt(M);

		opNNet.label 			= funcGrad;

		opNNet.epsilon				= 1e-5;
		opNNet.report_interval		= 10;
		opNNet.l2_penalty			= 1e-6;

		weakLearner = nNet(features,param0,opNNet,'regress','train');

		%clean up unnecessary parameters
		weakLearner.LabelHidden(abs(weakLearner.LabelHidden) < 1e-2*max(abs(weakLearner.LabelHidden(:)))) 		= 0;
		weakLearner.HiddenVisible(abs(weakLearner.HiddenVisible) < 1e-2*max(abs(weakLearner.HiddenVisible(:))))	= 0;

	elseif strcmpi(LearnerName,'deepNet') | strcmpi(LearnerName,'deepNNet')
	
		deepNetOpt.nIters				= 1000;
		deepNetOpt.l2_penalty			= 1e-4;
		deepNetOpt.epsilon				= 1e-6;
		deepNetOpt.report_interval		= 10;
		deepNetOpt.hiddenSizes			= [5,5]; %top down
		deepNetOpt.LabelSize			= L;
		deepNetOpt.activationType		= 'ReLU';
		
		if isfield(opt,'activationType')
			deepNetOpt.activationType 	= opt.activationType;
		end		

		if isfield(opt,'hiddenSizes')
			deepNetOpt.hiddenSizes		= opt.hiddenSizes;
		end

		deepNetOpt.label = funcGrad;
		weakLearner = deepNet(features,[],deepNetOpt,'regress', 'train');

	elseif strcmpi(LearnerName,'LeastSquare') | strcmpi(LearnerName,'Linear') | strcmpi(LearnerName,'RidgeRegress') | strcmpi(LearnerName,'Lasso') | strcmpi(LearnerName,'ElasticNet')

		if ~isfield(opt,'l1_penalty')
			%opt.l1_penalty 		= 1/(((abs(mean(funcGrad)) + std(funcGrad))^2)*trainSize); 
			%opt.l1_penalty 		= 1/((abs(mean(funcGrad)) + std(funcGrad))*trainSize); 
			opt.l1_penalty 		= 0.5/trainSize; 
		end
		
		opt.l2_penalty 		= 0/trainSize; 
		opt.epsilon			= 1e-5;
		opt.report_interval	= 10;
		opt.nIters			= 1000;

		if L==1
			weakLearner = regress([ones(trainSize,1),features],funcGrad,[],opt);
		else
			for l=1:L
				weakLearner{l} = regress([ones(trainSize,1),features],funcGrad(:,l),[],opt);
			end
		end

	elseif strcmpi(LearnerName,'PLS') %partial least square
	
		K = 10;
		
		if L==1
			[XL,yl,XS,YS,weakLearner] = plsregress(full(features),full(funcGrad),K);
		else
			for l=1:L
				[XL,yl,XS,YS,weakLearner{l}] = plsregress(full(features),full(funcGrad(:,l)),K);
			end
		end
		
	elseif strcmpi(LearnerName,'RegressTree')
	
		MinLeaf		= min(20,ceil(trainSize/64));
		if isfield(opt,'MinLeaf')
			MinLeaf = opt.MinLeaf;
		end
		NVarToSample = M; %default
		if isfield(opt,'NVarToSample')
			NVarToSample = opt.NVarToSample;
		end
	
		if L==1
			weakLearner = RegressionTree.fit(full(features),full(funcGrad),'MinLeaf',MinLeaf,'NVarToSample',NVarToSample);
			%weakLearner = fittree(full(features),full(funcGrad),'MinLeafSize',MinLeaf); %newer version, Matlab 2015?
		else
			for l=1:L
				weakLearner{l} = RegressionTree.fit(full(features),full(funcGrad(:,l)),'MinLeaf',MinLeaf,'NVarToSample',NVarToSample);
				%weakLearner{l} = fitrtree(full(features),full(funcGrad(:,l)),'MinLeaf',MinLeaf); %newer version, Matlab 2015?
			end
		end
	else
		
		fprintf('ERR: unknown learner name: %s\n',LearnerName);
		
	end

%- the functional gradient --
function [funcGrad,otherGrad] = getFuncGrad(funcVals,refOutcomes,Surrogate,opt)
	
	if strcmpi(Surrogate,'RMSE')
	
		funcGrad = refOutcomes-funcVals;
		
	elseif strcmpi(Surrogate,'MAE')
	
		funcGrad = sign(refOutcomes-funcVals);

	elseif strcmpi(Surrogate,'R2')
	
		[score,funcGrad] = calcR2(funcVals,refOutcomes);
		
	elseif strcmpi(Surrogate,'Logit')
	
		funcGrad = refOutcomes - 1./(1+exp(-funcVals));
		
	elseif strcmpi(Surrogate,'Maxent') | strcmpi(Surrogate,'softmax')
		
		L = size(funcVals,2);
		
		probs = exp(funcVals);
		probs = probs ./ (sum(probs,2)*ones(1,L));
		
		funcGrad = refOutcomes - probs;
		
	elseif strcmpi(Surrogate,'ListNet')
	
		probs = exp(funcVals);
		probs = probs ./ sum(probs);
		
		empiProbs = exp(refOutcomes);
		empiProbs = empiProbs ./ sum(empiProbs);
		
		funcGrad = (empiProbs - probs)*length(refOutcomes);
	elseif strcmpi(Surrogate,'SeqOrd') %sequential ordinal modelling
		
		%ordinal level starts from 1
		L = opt.MaxLevel;
	
		%the data id/level
		for l=1:L
			dataIds{l} = find(refOutcomes==l);
		end
		
		%CDF
		CDFs = ones(dataSize,L);
		for l=1:L-1
			CDFs(:,l) = 1./(1+exp(-(ones(dataSize,1)*opt.Threshold(l) - funcVals)));
		end
		
		% data likelihood
		probs = ones(dataSize,1);
		
		preCumuls = ones(dataSize,1);
		for l=1:L-1
			probs(dataIds{l}) = preCumuls(dataIds{l}) .* CDFs(dataIds{l},l);
			preCumuls = preCumuls .* (1-CDFs(:,l));
		end
		probs(dataIds{L}) = preCumuls(dataIds{L});
		
		%functional gradients
		funcGrad = zeros(dataSize,L-1);
		units = ones(dataSize,1);
		for l=1:L-1
			deltas = zeros(1,l);
			deltas(l) = 1;
			funcGrad(dataIds{l},1:l) = units(dataIds{l})*deltas - CDFs(dataIds{l},1:l);
		end	
		funcGrad(dataIds{L},1:L-1) = - CDFs(dataIds{L},1:L-1);
		
	elseif strcmpi(Surrogate,'CumulOrd') | strcmpi(Surrogate,'CumOrd') %cumulative ordinal modelling
		
		L = opt.MaxLevel;
		dataSize = length(refOutcomes);

		thresholds	= opt.Threshold;

		%CDFs = zeros(dataSize,L-1);
		%CDFs(:,1) 		= 1./(1+exp(-(ones(dataSize,1)*thresholds(1) - funcVals)));
		%CDFs(:,2:L-1)	= 1./(1+exp(-(ones(dataSize,1)*thresholds(2:L-1) - funcVals*ones(1,L-2))));

		CDFs	= 1./(1+exp(-(ones(dataSize,1)*thresholds - funcVals*ones(1,L-1))));

		probs = zeros(dataSize,1);
		probs(refOutcomes==1) = CDFs(refOutcomes==1,1);
		for l=2:L-1
			probs(refOutcomes==l) = CDFs(refOutcomes==l,l)-CDFs(refOutcomes==l,l-1);
		end
		probs(refOutcomes==L) = 1-CDFs(refOutcomes==L,L-1);

		funcGrad = zeros(dataSize,1);
		thresholdGrad = zeros(1,L-1);

		%the upper-bound
		for l=1:L-1
			funcGrad2 = CDFs(refOutcomes==l,l) .* (1-CDFs(refOutcomes==l,l)) ./ probs(refOutcomes==l);
			funcGrad(refOutcomes==l) = funcGrad(refOutcomes==l) - funcGrad2;
			thresholdGrad(l) = thresholdGrad(l) + sum(funcGrad2);
		end

		%the lower-bound
		for l=2:L
			funcGrad2 = CDFs(refOutcomes==l,l-1) .* (1-CDFs(refOutcomes==l,l-1)) ./ probs(refOutcomes==l);
			funcGrad(refOutcomes==l) = funcGrad(refOutcomes==l) + funcGrad2;
			thresholdGrad(l-1) = thresholdGrad(l-1) - sum(funcGrad2);
		end

		thresholdGrad(1) = 0;
		
		otherGrad = thresholdGrad ./ dataSize;
		
	elseif strcmpi(Surrogate,'PoissonRegress') %Poisson regression

		funcGrad = refOutcomes - exp(funcVals);
	end
	
	
%- the search direction --
function [searchDirection] = getSearchDirection(features,weakLearner,LearnerName,opt)

	[dataSize,M] = size(features);

	L = 1;
	if iscell(weakLearner)
		L = length(weakLearner);
	end

	if strcmpi(LearnerName,'NNet')
	
		searchDirection = nNet(features,weakLearner,opt,'regress','test');

	elseif strcmpi(LearnerName,'deepNet') | strcmpi(LearnerName,'deepNNet')

		searchDirection = deepNet(features,weakLearner,opt,'regress','test');
		
	elseif strcmpi(LearnerName,'LeastSquare') | strcmpi(LearnerName,'RidgeRegress') | strcmpi(LearnerName,'Lasso') | strcmpi(LearnerName,'ElasticNet') | strcmpi(LearnerName,'Linear')
		
		searchDirection = [ones(dataSize,1),features]*weakLearner';

	elseif strcmpi(LearnerName,'PLS') %partial least square
		
		searchDirection	= [ones(dataSize,1),features]*weakLearner;

	elseif strcmpi(LearnerName,'RegressTree')

		if L==1
			searchDirection = predict(weakLearner,full(features));
		else
			for l=1:L
				searchDirection(:,l) = predict(weakLearner{l},full(features));
			end
		end
	else
		
		fprintf('ERR: unknown learner name: %s\n',LearnerName);
	end

	searchDirection = tanh(searchDirection); %making things non-linear and robust against scales


%-----
function [probs] = getProbs(funcVals,Surrogate,opt)

	if strcmpi(Surrogate,'CumulOrd') | strcmpi(Surrogate,'CumOrd') %cumulative ordinal modelling
	
		L = length(opt.Threshold)+1;
		dataSize = length(funcVals);
		thresholds	= opt.Threshold;

		CRFs = zeros(dataSize,L-1);
		CDFs(:,1) 		= 1./(1+exp(-(ones(dataSize,1)*thresholds(1) - funcVals)));
		CDFs(:,2:L-1)	= 1./(1+exp(-(ones(dataSize,1)*thresholds(2:L-1) - funcVals*ones(1,L-2))));

		probs = zeros(dataSize,L);
		probs(:,1) 		= CDFs(:,1);
		probs(:,2:L-1)	= CDFs(:,2:L-1)-CDFs(:,1:L-2);
		probs(:,L) 		= 1-CDFs(:,L-1);
		
	elseif strcmpi(Surrogate,'binClassify')
	
		probs = 1 ./ (1+exp(-func));
		
	elseif strcmpi(Surrogate,'Maxent') | strcmpi(Surrogate,'multiClassify') | strcmpi(Surrogate,'softmax')
		
		L = size(funcVals,2);
		
		probs = exp(funcVals);
		probs = probs ./ (sum(probs,2)*ones(1,L));

	end