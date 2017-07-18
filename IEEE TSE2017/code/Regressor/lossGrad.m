function [ll,dGrad,thresholdGrad] = lossGrad(label,labelVals,opt)

epsilon = 1e-5; %smoothing of probabilities - avoiding the numerical problems

[dataSize,L] = size(label);

thresholdGrad = [];
if strcmpi(opt.outputMode,'softmax')
	labelProbs	= exp(labelVals);
	Zs			= sum(labelProbs,2);
	labelProbs	= labelProbs ./ (Zs * ones(1,opt.LabelSize));
	
	%smoothing
	labelProbs = (1-epsilon)*labelProbs + epsilon/opt.LabelSize;

	empiProbs = label;

	ll = sum(log(labelProbs(label==1)));

	%backpropagation
	if nargout >= 2
		dGrad = empiProbs - labelProbs;
	end
elseif strcmpi(opt.outputMode,'logit')
	labelProbs		= 1./ (1 + exp(-labelVals));

	%smoothing
	labelProbs = (1-epsilon)*labelProbs + epsilon*0.5;
	
	lls = label.*log(labelProbs) + (1-label).*log(1-labelProbs);
	
	if isfield(opt,'labelWeight')
		lls = lls .* repmat(opt.labelWeight,dataSize,1);
	end
	
	if isfield(opt,'weight')
		lls = lls .* opt.weight;
	end	
	
	ll = sum(lls(:));

	if nargout >= 2
		dGrad	= label - labelProbs;

		if isfield(opt,'labelWeight')
			dGrad = dGrad .* repmat(opt.labelWeight,dataSize,1);
		end

		if isfield(opt,'weight')
			dGrad = dGrad .* opt.weight;
		end	
	end
elseif strcmpi(opt.outputMode,'regress') | strcmpi(opt.outputMode,'regress-mean')
	
	err		= label - labelVals;
	ll 		= -0.5*sum(err(:).^2);

	if nargout >= 2
		dGrad 	= err;
	end

elseif strcmpi(opt.outputMode,'regress-median')
	
	err		= label - labelVals;
	MAE		= sqrt(1e-3 + err.*err);

	ll 		= -sum(MAE(:));

	if nargout >= 2
		dGrad 	= err ./ MAE;
	end

elseif strcmpi(opt.outputMode,'cumOrd')

	L2 = opt.opt.LabelSize;

	CDFs = 1./(1+exp(-(repmat(thresholds,dataSize,1) - repmat(labelVals,1,L2-1))));

	probs = zeros(dataSize,1);
	probs(label==1) = CDFs(label==1,1);
	for l=2:L2-1
		probs(label==l) = CDFs(label==l,l)-CDFs(label==l,l-1);
	end
	probs(label==L2) = 1-CDFs(label==L2,L2-1);

	ll = sum(log(1e-10 + probs));

	if nargout >= 2

		dGrad = zeros(dataSize,1);
		thresholdGrad2 = zeros(1,L2-1);

		%the upper-bound
		for l=1:L2-1

			dGrad2 = CDFs(label==l,l) .* (1-CDFs(label==l,l));
			dGrad2 = dGrad2 ./ probs(label==l);

			dGrad(label==l) = dGrad(label==l) - dGrad2;
			thresholdGrad2(l) = thresholdGrad2(l) + sum(dGrad2);
		end

		%the lower-bound
		for l=2:L2
			dGrad2 = CDFs(label==l,l-1) .* (1-CDFs(label==l,l-1));
			dGrad2 = dGrad2 ./ probs(label==l);

			dGrad(label==l) = dGrad(label==l) + dGrad2;
			thresholdGrad2(l-1) = thresholdGrad2(l-1) - sum(dGrad2);
		end

		thresholdGrad 		= zeros(1,L2-1);
		thresholdGrad(1)	= sum(thresholdGrad2);
		for l=2:L2-1
			thresholdGrad(l) = sum(thresholdGrad2(l:L2-1))*thresholds(l);
		end
		thresholdGrad = thresholdGrad ./ dataSize;
		thresholdGrad(1) = 0;
	end
end

%task regularization, not its parameters
%if isfield(opt,'TaskCorrel') & ~isempty(opt.TaskCorrel) & isfield(opt,'task_correl_penalty') & opt.task_correl_penalty
%	ll 		= ll  - 0.5*opt.task_correl_penalty*sum(sum((labelVals*opt.TaskCorrel).*labelVals));
%	if nargout >= 2
%		dGrad	= dGrad  - opt.task_correl_penalty*(opt.TaskCorrel*labelVals')';
%	end
%end
