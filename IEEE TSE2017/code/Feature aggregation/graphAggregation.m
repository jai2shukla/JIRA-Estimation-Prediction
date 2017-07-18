function [aggregatedFeatures] = graphAggregation(baseFeatures, subFeatures, graphData,numId)
%this function extracts features of graph from sprint using pair of issues
%Feature of graph:  number of node 
%                   number of edge
%                   sum of fan in (sum of fan in edge)The number of inward directed graph edges from a given graph vertex in a directed graph. 
%                   min of fan in
%                   max of fan in
%                   Avg of fan in (sum of fan in edge/number of edge)
%                   mode of fan in
%                   
%                   sum, avg, min, max, mode of fan out (sum of fan out edge) The number of outward directed graph edges from a given graph vertex in a directed graph. 
%                   Avg. Node degree = 2E/N
    
    aggregatedFeatures = unique(baseFeatures);
    FeatureGraph = double(aggregatedFeatures(:,1:numId));
    for v=1:size(aggregatedFeatures,1)
        graphInSprint = graphData(graphData.(1)==FeatureGraph(v,1)&graphData.(2)==FeatureGraph(v,2),:);
        
        noNode = 0;
        noEdge = 0;
        
        sumFanOut = 0;
        minFanOut = 0;
        maxFanOut = 0;
        meanFanOut = 0;
        modeFanOut = 0;
        
        sumFanIn = 0;
        minFanIn = 0;
        maxFanIn = 0;
        meanFanIn = 0;
        modeFanIn = 0;
        
        nodeDegree = 0;
        
        if size(graphInSprint,1)
            noNode = size(unique([graphInSprint.(3) graphInSprint.(4)]),1);
            noEdge = size(graphInSprint,1);
            
            fanOut = tabulate(graphInSprint.(3));
            fanOut = cell2mat(fanOut(:,2));
            
            sumFanOut = sum(fanOut);
            minFanOut = min(fanOut);
            maxFanOut = max(fanOut);
            meanFanOut = mean(fanOut);
            modeFanOut = mode(fanOut);
            
            fanIn = tabulate(graphInSprint.(4));
            fanIn = cell2mat(fanIn(:,2));
            
            sumFanIn = sum(fanIn);
            minFanIn = min(fanIn);
            maxFanIn = max(fanIn);
            meanFanIn = mean(fanIn);
            modeFanIn = mode(fanIn);
            
            nodeDegree = (2 * noEdge) / noNode;
        end
        FeatureGraph(v,3) = noNode;
        FeatureGraph(v,4) = noEdge;
        FeatureGraph(v,5) = sumFanOut;
        FeatureGraph(v,6) = minFanOut;
        FeatureGraph(v,7) = maxFanOut;
        FeatureGraph(v,8) = meanFanOut;
        FeatureGraph(v,9) = modeFanOut;
        FeatureGraph(v,10) = sumFanIn;
        FeatureGraph(v,11) = minFanIn;
        FeatureGraph(v,12) = maxFanIn;
        FeatureGraph(v,13) = meanFanIn;
        FeatureGraph(v,14) = modeFanIn;
        FeatureGraph(v,15) = nodeDegree;
    end
    catName = {'boardid','sprintid','noNode','noEdge','sumFanOut','minFanOut','maxFanOut','meanFanOut','modeFanOut','sumFanIn','minFanIn','maxFanIn','meanFanIn','modeFanIn','nodeDegree'};
    FeatureGraphDS = mat2dataset(FeatureGraph,'VarNames',catName);
    aggregatedFeatures = join(aggregatedFeatures,FeatureGraphDS,'Keys',catName(1:2));
end