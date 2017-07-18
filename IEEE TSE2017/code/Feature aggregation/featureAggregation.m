function [ aggregatedFeatures ] = featureAggregation(baseFeatures, subFeatures, numId)
%This function aggregates feature based on Feature aggregration method.
% When we have two level of features.
%   Input:  1. features = rawdata table which consists of base features and
%   a set of features that need to aggregrate (sprint-issue).
%               baseFeatures, subFeatures (will be aggregated)
%           2. number of base attribute that use for grouping.
%           3. indexOfSubFeature = the start index of subfeature
%           4. numId = Number of identification column (identical column)
%           4. Aggregation Method (simple: simple statistic)
%   Output: this function produces a aggregated data table
%           (labels, based features, arrgreated feature) ready for
%           classifier (categorial is encode to binary)

    aggregatedFeatures = unique(baseFeatures);
    [numSubElement, numSubFeature] = size(subFeatures);
    [numAggElement, numAggFeature] = size(aggregatedFeatures);
    for v = 1:numId
        idName(v) = baseFeatures.Properties.VarNames(v);
    end

    if 1 
        for v = numId+1 : numSubFeature
            if (isnumeric(subFeatures.(v))) % numeric value: calculate simple satistic and concat to aggregatedFeatures
                featureStat = grpstats(subFeatures, idName,{'min','max','std','mean','median','var','range'},'DataVars',subFeatures.Properties.VarNames(v));
                featureStat(:,'GroupCount') = [];
                aggregatedFeatures = join(aggregatedFeatures,featureStat,'Keys',idName);
            else %categorical value: count frequency of category.
                 %find all cat of column v
                 subFeatures.(v) = regexprep((subFeatures.(v)).','[^\w'']','').';
                 catName = unique(subFeatures.(v)).';
                 %catTable = cell2dataset(cell(numAggElement,length(catName)),'VarNames',catName);
                 %Unique basevalue
                 [frequency index member] = unique(baseFeatures(:,1:numId));
                 %frequency = horzcat(frequency ,catTable);
                 
                 catElement = subFeatures(:,[1:numId v]);
                 catElement.(3) = categorical(catElement.(3));
                 catElement.(4) =  member(:,1);
                 
                 for i=1:numAggElement
                    catValue = catElement(catElement.(4)==i, 3);
                    count = tabulate(catValue.(1));
                    countTable(i,:) = count(:,2).';
                 end
                 countTable = cell2mat(countTable);
                 frequency = horzcat(frequency, mat2dataset(countTable,'VarNames',catName));
                 aggregatedFeatures = join(aggregatedFeatures,frequency,'Keys',idName);
                 clear countTable;
            end
        end
    end
end

