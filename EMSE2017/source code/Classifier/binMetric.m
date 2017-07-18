function [Rec,Prec,F1,AUC] = binMetric(actual,predict,prob)

	if isempty(predict)
		predict = (prob >= 0.5);
	end

	Rec = sum(actual == 1 & predict == 1) / (1e-5 + sum(actual == 1));
	Prec = sum(actual == 1 & predict == 1) / (1e-5 + sum(predict == 1));
	F1 = 2*Rec*Prec/(1e-5 + Rec + Prec);
	[AUC,CI] = auc([actual,prob],0.05,'mann-whitney');
