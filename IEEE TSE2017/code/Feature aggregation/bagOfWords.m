function  [ aggregatedFeatures ] = bagOfWords(baseFeatures, subFeatures, numId, k)
% This function aggregate a set of features using Kmeans. Then calculate
% BOW
%Bag-of-Words (BOW)
%1. First we stack all sets of vectors to create a matrix.
%2. Then run k-means to find out k cluster centers. Typically k = 10; 20; 50; 100; 200; 500.
%3. For each vector find the center c.
%4. Then, for each vector set, for each center c, count the number of times the
%   center is associated with. For each set, we will then have a bag-of-words
%   (BOW), where each word is the center index. The BOWis the new feature
%   vector for the set.
%This function outputs a aggregated features (date set type) for classificier

% get id name
aggregatedFeatures = unique(baseFeatures);
[numSubElement, numSubFeature] = size(subFeatures);
[numAggElement, numAggFeature] = size(aggregatedFeatures);
for v = 1:numId
    idName(v) = baseFeatures.Properties.VarNames(v);
end

% change non numeric value to binary features
if 1
    %binary cat
    for v = numId+1 : numSubFeature
        if(~isnumeric(subFeatures.(v)))
            subFeatures.(v) = regexprep((subFeatures.(v)).','[^\w'']','').';
            catName = unique(subFeatures.(v)).';
            catName{length(catName)+1} = 'Null';
            subFeatures = categorical2bins(subFeatures,cell2mat(subFeatures.Properties.VarNames(v)),catName);
        end
    end
    %remove cat column
    v = numId;
    while v <= size(subFeatures,2)
        if(~isnumeric(subFeatures.(v)))
            subFeatures(:,subFeatures.Properties.VarNames(v)) = [];
        else
            v = v + 1;
        end
    end
    % call k means function (k cluster)
    [IDX, center] = kmeans(double(subFeatures(:,numId+1:end)),k,'Display','iter');
    %%
    % count frequency of vector associated to k center to generate bag
    % of features
    [frequency,index,member] = unique(baseFeatures(:,1:numId));
    clusterName = unique(IDX).';
    clusterElement = subFeatures(:,[1:numId]);
    clusterElement.(3) = categorical(IDX);
    clusterElement.(4) = member(:,1);
    
    for i=1:numAggElement
        catValue = clusterElement(clusterElement.(4)==i, 3);
        count = tabulate(catValue.(1));
        cluster(i,:) = count(:,2).';
    end
    frequency = horzcat(frequency, mat2dataset(cell2mat(cluster)));
    aggregatedFeatures = join(aggregatedFeatures,frequency,'Keys',idName);
end
end