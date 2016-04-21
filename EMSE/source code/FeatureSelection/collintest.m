function varargout = collintest(X,varargin)
%COLLINTEST Belsley collinearity diagnostics
%
% Syntax:
%
%   [sValue,condIdx,VarDecomp] = collintest(X)
%   [sValue,condIdx,VarDecomp] = collintest(X,param,val,...)
%
% Description:
%
%   COLLINTEST assesses the strength and sources of collinearity among
%   variables in a multiple linear regression of the form y = X*b + e.
%   Singular values of scaled X are converted to condition indices, which
%   identify the number and strength of any near dependencies in the data.
%   The variance of the OLS estimate of b is decomposed in terms of the
%   singular values to identify variables involved in each near dependency,
%   as well as the extent to which the dependencies degrade the regression.
%
% Input Arguments:
%
%   X - numObs-by-numVars matrix or tabular array of variables to be used
%       in a multiple linear regression. For models with an intercept, X
%       should contain a column of ones. Data in X should not be centered.
%       Columns of X are scaled to unit length before processing.
%
% Optional Input Parameter Name/Value Pairs:
%
%   NAME        VALUE
%
%   'varNames' 	Cell vector of variable name strings of length numVars to
%               be used in displays and plots of the results. Names should
%               include the intercept term (for example, 'Const'), if
%               present. The default for matrix X is {'var1','var2',...}.
%               The default for tabular array X is
%               X.Properties.VariableNames.
%
%	'display'   String indicating whether or not to display results to the
%               command window. The display shows all outputs in tabular
%               form. Values are 'on' and 'off'. The default is 'on'.
%
%	'plot'      String indicating whether or not to plot the results. The
%               plot shows the critical rows of the output VarDecomp whose
%               condition indices are above the input tolerance tolIdx.
%               Groups of variables with variance-decomposition proportions
%               above the input tolerance tolProp are identified with red
%               markers. Values are 'on' and 'off'. The default is 'off'.
%
%	'tolIdx'    Scalar tolerance on the condition indices, used to decide
%               which indices are large enough to infer a near dependency
%               in the data. The value is only used if 'plot' is 'on'. The
%               value must be at least 1. The default is 30.
%
%	'tolProp'   Scalar tolerance on the variance-decomposition proportions,
%               used to decide which variables are involved in any near
%               dependency. The value is only used if 'plot' is 'on'. The
%               value must be between 0 and 1. The default is 0.5.
%
% Output Arguments:
%
%   sValue - Vector of singular values of the scaled design matrix X, in
%       descending order. sValue is the ordered diagonal of the matrix S in
%       the singular-value decomposition U*S*V' of scaled X.
%
%   condIdx - Vector of condition indices sValue(1)/sValue(j),
%       j = 1, ..., numVars, in ascending order. Large indices identify
%       near dependencies among the variables in X. The size of the indices
%       is a measure of how near dependencies are to collinearity.
%
%   VarDecomp - numVars-by-numVars array of variance-decomposition
%       proportions. The variance of the estimate of ith regression
%       parameter b(i) is proportional to the sum:
%       
%           V(i,1)^2/sValue(1)^2 + ... + V(i,numVars)^2/sValue(numVars)^2
%       
%       where V is the matrix of orthonormal eigenvectors of X'*X obtained
%       from the singular-value decomposition U*S*V' of scaled X.
%       VarDecomp(i,j) is the proportion of the jth term in the sum
%       relative to the entire sum. Large proportions, combined with a
%       large condition index, identify groups of variates involved in near
%       dependencies. The size of the proportions is a measure of how badly
%       the regression is degraded by the dependency.
%
% Notes:
%
%   o The condition number of scaled X is sValue(1)/sValue(numVars). All
%     condition indices in condIdx are thus between 1 and the condition
%     number. The condition number achieves its lower bound of 1 when the
%     columns of scaled X are orthonormal, and rises as variates exhibit
%     greater dependency. The condition number is often used as an overall
%     diagnostic for detecting collinearity, though it fails to provide
%     specifics on the strength and sources of any near dependencies.
%
%   o The expressions sValue(j)^2 found in the variance decomposition are
%     the eigenvalues of scaled X'*X. Thus large variance-decomposition
%     proportions correspond to small eigenvalues of X'*X, a common
%     diagnostic. The singular-value decomposition provides a more direct,
%     numerically stable view of the eigensystem of scaled X'*X.
%
%   o For purposes of collinearity diagnostics, Belsley [1] shows that
%     column scaling of X is always desirable. However, centering the data
%     in X is undesirable. For models with an intercept, centering can hide
%     the role of the constant term in any near dependency and produce
%     misleading diagnostics.
%
%   o Tolerances for "high" condition indices and variance-decomposition
%     proportions are comparable to critical values in standard hypothesis
%     tests. Experience determines the most useful tolerances, but
%     experiments [1] suggest COLLINTEST defaults as a good starting point.
%
% Example:
%
%   load Data_Canada
%   collintest(DataTable,'plot','on')
%
% References:
% 
%   [1] Belsley, D. A., E. Kuh, and R. E. Welsh. Regression Diagnostics.
%       New York, NY: John Wiley & Sons, Inc., 1980.
% 
%   [2] Judge, G. G., W. E. Griffiths, R. C. Hill, H. Lutkepohl, and T. C.
%       Lee. The Theory and Practice of Econometrics. New York, NY: John
%       Wiley & Sons, Inc., 1985.
%
% See also CORRPLOT.

% Copyright 2012 The MathWorks, Inc.

% Handle dataset array inputs:

if isa(X,'dataset')
    
    try
    
        X = dataset2table(X);
    
    catch 
    
        error(message('econ:collintest:DataNotConvertible'))
    
    end
    
end

% Parse inputs and set defaults:

parseObj = inputParser;
parseObj.addRequired('X',@XCheck);
parseObj.addParamValue('varNames',{},@varNamesCheck);
parseObj.addParamValue('display','on',@displayCheck);
parseObj.addParamValue('plot','off',@plotCheck);
parseObj.addParamValue('tolIdx',30,@tolIdxCheck);
parseObj.addParamValue('tolProp',0.5,@tolPropCheck);

parseObj.parse(X,varargin{:});

X = parseObj.Results.X;
varNames = parseObj.Results.varNames;
displayFlag = strcmpi(parseObj.Results.display,'on');
plotFlag = strcmpi(parseObj.Results.plot,'on');
tolIdx = parseObj.Results.tolIdx;
tolProp = parseObj.Results.tolProp;

[numObs,numVars] = size(X);

% Create variable names:

if displayFlag || plotFlag

    if isempty(varNames)

        if isa(X,'table')

            varNames = X.Properties.VariableNames;

        else

            varNames = strcat({'var'},num2str((1:numVars)','%-u'));

        end

    else

        if length(varNames) < numVars

            error(message('econ:collintest:VarNamesTooFew'))

        elseif length(varNames) > numVars

            error(message('econ:collintest:VarNamesTooMany'))

        end

    end
    
end

% Convert table to double for numeric processing:

if isa(X,'table')
    
    try
    
        X = table2array(X);
        X = double(X);
    
    catch 
    
        error(message('econ:collintest:DataNotConvertible'))
    
    end
    
end

% Scale columns to length 1:

colNormsX = sqrt(sum(X.^2));
colNormsX(colNormsX == 0) = 1; % Avoid divide by 0
XS = X./repmat(colNormsX,numObs,1); % Scaled X

% Compute SVD:

[~,S,V] = svd(XS,0);
sValue = diag(S);

% Compute condition indices:

sValue(sValue < eps) = eps; % Avoid divide by 0
condIdx = sValue(1)./sValue;

% Compute variance decomposition proportions:

PHI = (V.^2)./repmat((sValue.^2)',numVars,1);
phi = sum(PHI,2);
VarDecomp = PHI'./repmat(phi',numVars,1);

% Display diagnostic information:

if displayFlag

    fprintf('\nVariance Decomposition\n\n')
    internal.econ.tableprint([sValue,condIdx,VarDecomp],...
                             'colNames',[{'sValue','condIdx'},varNames{:}])
    
end

% Plot diagnostic information:

if plotFlag
    
    criticalIdx = (condIdx > tolIdx);
    numCritical = sum(criticalIdx);
    CriticalVarDecomp = VarDecomp(criticalIdx,:);
    CriticalProp = (CriticalVarDecomp > tolProp);
    
    if numCritical == 0
        
        warning(message('econ:collintest:NoCriticalRows'))
        return   
            
    end
    
    hFig = figure('Tag','collintestFigure');
    hold on
    plotVars = 1:numVars;
    colors = jet(numCritical);
    hRows = zeros(numCritical,1);
  
    for i = 1:numCritical
    
        hRows(i) = plot(plotVars,CriticalVarDecomp(i,:),...
                        '-o','LineWidth',2,'Color',colors(i,:),...
                        'Tag','decompLines');
        
        % Mark proportions in red if at least two are critical:
        if sum(CriticalProp(i,:)) > 1
            
            plot(plotVars(CriticalProp(i,:)),...
                 CriticalVarDecomp(i,CriticalProp(i,:)),...
                 'o','MarkerFaceColor','r','Tag','criticalPoints')
             
        end
        
    end
    
    hTol = plot(1:numVars,repmat(tolProp,numVars,1),'--m','LineWidth',2,...
        'Tag','tolPropLine');
    
    xlim([1 numVars])
    hAxes = get(hFig,'CurrentAxes');
    set(hAxes,'XTick',1:numVars)
    set(hAxes,'XTickLabel',varNames)
    xlabel(hAxes,'Variables')
    ylabel(hAxes,'Variance-Decomposition Proportions')
    title(hAxes,'{\bf High Index Variance Decompositions}')
    legendLabels = strcat({'condIdx '},num2str(condIdx(criticalIdx),'%-.3g'));
    legend(hAxes,[hRows;hTol],[legendLabels;{'tolProp'}],'Location','Best')
    grid on
    hold off
    
end

nargoutchk(0,3);

if nargout > 0
    
    varargout = {sValue,condIdx,VarDecomp};
    
end

%-------------------------------------------------------------------------
% Check input X
function OK = XCheck(X)

if ischar(X)
    
    error(message('econ:collintest:DataNonNumeric'))
            
elseif isempty(X)

    error(message('econ:collintest:DataUnspecified'))

elseif isvector(X)

    error(message('econ:collintest:DataIsVector'))

else

    OK = true;

end

%-------------------------------------------------------------------------
% Check value of 'varNames' parameter
function OK = varNamesCheck(varNames)
    
if ~isvector(varNames)

    error(message('econ:collintest:VarNamesNonVector'))

elseif isnumeric(varNames) || (iscell(varNames) && any(cellfun(@isnumeric,varNames)))

    error(message('econ:collintest:VarNamesNumeric'))

else

    OK = true;

end

%-------------------------------------------------------------------------
% Check value of 'display' parameter
function OK = displayCheck(displayParam)
    
if ~isvector(displayParam)

    error(message('econ:collintest:DisplayParamNonVector'))

elseif isnumeric(displayParam)

    error(message('econ:collintest:DisplayParamNumeric'))

elseif ~ismember(lower(displayParam),{'off','on'})

    error(message('econ:collintest:DisplayParamInvalid'))

else

    OK = true;

end

%-------------------------------------------------------------------------
% Check value of 'plot' parameter
function OK = plotCheck(plotParam)
    
if ~isvector(plotParam)

    error(message('econ:collintest:PlotParamNonVector'))

elseif isnumeric(plotParam)

    error(message('econ:collintest:PlotParamNumeric'))

elseif ~ismember(lower(plotParam),{'off','on'})

    error(message('econ:collintest:PlotParamInvalid'))

else

    OK = true;

end

%-------------------------------------------------------------------------
% Check value of 'tolIdx' parameter
function OK = tolIdxCheck(tolIdx)
    
if ~isscalar(tolIdx)

    error(message('econ:collintest:TolIdxNonScalar'))

elseif ~isnumeric(tolIdx)

    error(message('econ:collintest:TolIdxNonNumeric'))
      
elseif tolIdx < 1
        
	error(message('econ:collintest:TolIdxOutOfRange'))

else

    OK = true;

end

%-------------------------------------------------------------------------
% Check value of 'tolProp' parameter
function OK = tolPropCheck(tolProp)
    
if ~isscalar(tolProp)

    error(message('econ:collintest:TolPropNonScalar'))

elseif ~isnumeric(tolProp)

    error(message('econ:collintest:TolPropNonNumeric'))
      
elseif (tolProp < 0) || (tolProp > 1)
        
	error(message('econ:collintest:TolPropOutOfRange'))

else

    OK = true;

end