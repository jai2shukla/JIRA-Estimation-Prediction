% 14 July 2014: Morakot modified: add p-value method

function [selectIds,ignoreIds,selectFeatNames] = featSelect(features,labels,featNames,SelectionMethod)

	% binary labels
	
	[dataSize,N] = size(features);
	
	%-- TRAIN/VALID SPLITTING ---
    
	validIds = [1:5:dataSize]; % create validation set (each of 5)
	trainIds = setdiff([1:dataSize],validIds); % create training set using diff between validation set

	validFeatures	= features(validIds,:);
	trainFeatures	= features(trainIds,:);
	validLabels 	= labels(validIds);	
	trainLabels 	= labels(trainIds);	

	validSize		= length(validLabels);
	trainSize		= length(trainLabels);
	

    %SelectionMethod = 'l1_penalty'
    %SelectionMethod = 'p_value'
    
    if strcmp(SelectionMethod,'l1_penalty')
        model = 'logit';
        modelOpt.label 		= trainLabels;

        % first-round
        maxAUC = 0;
        best_l1_penalty = 0;
        for l1_penalty = [0.1 0.3 1 3 10]*1e-4
            modelOpt.l1_penalty = l1_penalty;
            glmParam2 	= glmBin(model,[trainFeatures,ones(trainSize,1)],zeros(1,N+1),modelOpt,'train');

            validProbs	= glmBin(model,[validFeatures,ones(validSize,1)],glmParam2,modelOpt,'test');
            [validAUC,validAucCI]	= auc([validLabels,validProbs],0.05,'mann-whitney');

            if validAUC > maxAUC
                maxAUC = validAUC;
                best_l1_penalty = l1_penalty;
            end
        end

        % second-round
        for l1_penalty = [0.4 0.5 0.7 1.3 1.8 2.5]*best_l1_penalty
            modelOpt.l1_penalty = l1_penalty;
            glmParam2 	= glmBin(model,[trainFeatures,ones(trainSize,1)],zeros(1,N+1),modelOpt,'train');

            validProbs	= glmBin(model,[validFeatures,ones(validSize,1)],glmParam2,modelOpt,'test');
            [validAUC,validAucCI]	= auc([validLabels,validProbs],0.05,'mann-whitney');

            if validAUC > maxAUC
                maxAUC = validAUC;
                best_l1_penalty = l1_penalty;
            end
        end

        modelOpt.l1_penalty = best_l1_penalty;
        glmParam2 	= glmBin(model,[trainFeatures,ones(trainSize,1)],zeros(1,N+1),modelOpt,'train');

        selectIds = find(abs(glmParam2(1:N)) >= 1e-3);
        ignoreIds = setdiff([1:N],selectIds);
        glmParam2(ignoreIds) = 0;
        
        j = 1;
        for i=1:N
            fprintf('%.3f\t%s\n',glmParam2(i),featNames{i});
            if abs(glmParam2(i)) >= 1e-3
                selectFeatNames{j} = featNames{i};
                j = j+1;
            end
        end
    end
    
    if strcmp(SelectionMethod,'p_value')
        model = 'binomial';
        p_value = NaN([1 N]);
        B = NaN([1 N]);
        for i=1:N
            [b,dev,stats] = glmfit(features(:,i),labels,model);   
            p_value(i) = stats.p(2);
            B(i) = b(2);
        end
        
        j=1;
        for i=1:N
            fprintf('%.3f\t%.3f\t%s\n',B(i),p_value(i),featNames{i});    
            if (p_value(i) < 0.05 && ~isnan(p_value(i)))
                selectFeatNames{j} = featNames{i};
                j = j+1;
            end
        end
        
        [sValue,condIdx,VarDecomp] = collintest(features(:,p_value < 0.05),'varName',selectFeatNames);
        p_value = p_value(p_value < 0.05);
        VarDecomp = VarDecomp(end,:);
        minP = min(p_value(VarDecomp >= 0.5))
        
        selectFeatNames = selectFeatNames(VarDecomp <= 0.5 | (VarDecomp > 0.5 & p_value == minP));
        % add dummy output
        selectIds = 0;
        ignoreIds = 0;
    end
end
	%best_l1_penalty
