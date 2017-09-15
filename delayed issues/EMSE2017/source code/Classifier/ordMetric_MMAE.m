function [NMAE] = ordMetric_MMAE(actual,predict,probs)
    NMAE = 0;
    L = max(actual);
    if isempty(predict)
		[val,predict] = sort(probs');
		predict = predict';
        predict = predict(:,end);
	end
    for l=1:L
		n = 1e-5  + sum(actual==l);
		NMAE = NMAE + sum(abs(l - predict(actual==l)))/n;
    end
	NMAE = NMAE/L;
end

