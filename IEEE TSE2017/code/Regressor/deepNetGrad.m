function [ll,dl] = deepNetGrad(param,opt)

% Deep neural network gradient, output is softmax | logistic regression

hiddenSizes	= opt.hiddenSizes;
[dataSize,N] = size(opt.data);
L = size(opt.label,2);
Depth = opt.Depth;

activationType = 'sigm';
if isfield(opt,'activationType')
	activationType = opt.activationType;
end

if ~isfield(opt,'activationTypes')
	for depth=1:Depth
		activationTypes{depth} = activationType;
	end
else
	activationTypes = opt.activationTypes;
end


% -- reshape parameters --
K = hiddenSizes(1);
labelParam			= param(1:L); lastF = L;
labelHiddenParam	= reshape(param(lastF+1:lastF+L*K),L,K); lastF = lastF+L*K;

for depth=1:Depth
	if depth < Depth
		K2 = hiddenSizes(depth+1);
	else
		K2 = N;
	end
	
	hiddenParam{depth}		= param(lastF+1:lastF+K); lastF = lastF+K;
	hiddenMapParam{depth}	= reshape(param(lastF+1:lastF+K*K2),K,K2); lastF = lastF+K*K2;
	
	K = K2;
end
% -- END reshape parameters --


%-- forward-propagation --
for depth=Depth:-1:1
	if depth == Depth
		hiddenVals = ones(dataSize,1)*hiddenParam{depth} + opt.data*hiddenMapParam{depth}';
	else
		hiddenVals = ones(dataSize,1)*hiddenParam{depth} +  hiddenActivations{depth+1}*hiddenMapParam{depth}';
	end
	
	hiddenActivations{depth} = getActivation(hiddenVals,activationTypes{depth});
	
	if isfield(opt,'dropOut') & ~isempty(opt.dropOut)
		hiddenActivations{depth}(opt.dropOut{depth}) = 0;
	end
	
	if isfield(opt,'selectDropRate') & opt.selectDropRate
		hiddenActivations{depth}(hiddenActivations{depth} < prctile(hiddenActivations{depth}(:),opt.selectDropRate*100)) = 0;
	end
	
end

labelVals = ones(dataSize,1)*labelParam + hiddenActivations{1}*labelHiddenParam';

[ll,dGrad,thresholdGrad] = lossGrad(opt.label,labelVals,opt);

ll = ll/dataSize - 0.5*opt.l2_penalty*param*param';

if isfield(opt,'noLabelBias') & opt.noLabelBias == 1;
	labelGrad	= zeros(1,L);
else
	labelGrad	= mean(dGrad,1);
end

%-- backward-propagation --

labelHiddenGrad	= dGrad'*hiddenActivations{1} ./ dataSize;

if isfield(opt,'gradNorm') & opt.gradNorm
	if norm(labelHiddenGrad(:)) < opt.gradNorm
		labelHiddenGrad = labelHiddenGrad * 0.1*opt.gradNorm./(1e-10 + norm(labelHiddenGrad(:))); %handle vanishing/exploding gradients
	end
end

propGrad = (dGrad*labelHiddenParam) .* getActivationGrad(hiddenActivations{1},activationTypes{1});

for depth=1:Depth-1
	hiddenGrad{depth}		= mean(propGrad,1);
	hiddenMapGrad{depth}	= propGrad'*hiddenActivations{depth+1} ./ dataSize;


	if isfield(opt,'gradNorm') & opt.gradNorm
		if norm(hiddenMapGrad{depth}(:)) < opt.gradNorm
			hiddenMapGrad{depth}	= hiddenMapGrad{depth} * opt.gradNorm/(1e-10 + norm(hiddenMapGrad{depth}(:)));  %handle vanishing/exploding gradients
		end
	end
	
	propGrad = (propGrad*hiddenMapParam{depth}) .* getActivationGrad(hiddenActivations{depth+1},activationTypes{depth+1});
end

hiddenGrad{Depth}		= mean(propGrad,1);
hiddenMapGrad{Depth}	= propGrad'*opt.data ./ dataSize;

if isfield(opt,'gradNorm') & opt.gradNorm
	if norm(hiddenMapGrad{Depth}(:)) < opt.gradNorm
		hiddenMapGrad{Depth}	= hiddenMapGrad{Depth} * opt.gradNorm/(1e-10 + norm(hiddenMapGrad{Depth}(:)));  %handle vanishing/exploding gradients
	end
end

%--- Parameter regularization ---

%controlling the norm
if opt.norm_penalty
	for depth=1:Depth
		if depth < Depth
			K2 = hiddenSizes(depth+1);
		else
			K2 = N;
		end

		hiddenNorms = sqrt(sum(hiddenMapParam{depth}.^2,2));

		ids 	= (hiddenNorms >= opt.MaxNorm);
		ids2	= repmat(ids,1,K2);

		diffNorms = hiddenNorms - opt.MaxNorm;
		ll = ll - 0.5*opt.norm_penalty*sum(diffNorms(ids).^2);

		normGrad = hiddenMapParam{depth}.*repmat((diffNorms./hiddenNorms),1,K2);
		hiddenMapGrad{depth}(ids2) = hiddenMapGrad{depth}(ids2) - opt.norm_penalty*normGrad(ids2);
	end
end	

%--- reshape the gradient ---

K = hiddenSizes(1);
grad = [labelGrad,reshape(labelHiddenGrad,1,L*K)];
for depth=1:Depth
	if depth < Depth
		K2 = hiddenSizes(depth+1);
	else
		K2 = N;
	end
	grad = [grad,hiddenGrad{depth},reshape(hiddenMapGrad{depth},1,K*K2)];
	
	K = K2;
end
%--- END reshape the gradient ---

dl = grad - opt.l2_penalty*param;


%----------------
