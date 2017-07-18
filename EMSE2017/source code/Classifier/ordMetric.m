function [NMAE,NMCE] = ordMetric(actual,predict,probs,costs)

	% probs: [N,L] the probabilities
	% costs: [1,L] the cost structure

	L = max(actual);

	expCosts = probs*costs';

	if isempty(predict)
		[val,predict] = sort(probs');
		predict = predict';
        predict = predict(:,end);
	end

	NMAE = 0;
	NMCE = 0;
	for l=1:L
		n = 1e-5  + sum(actual==l);
		NMAE = NMAE + sum(abs(l - predict(actual==l)))/n;
		NMCE = NMCE + sum(abs(costs(l) - expCosts(actual==l)))/n;
	end
	NMAE = NMAE/L;
	NMCE = NMCE/L;
