function [bb,dev,stats] = glmfit(x,y,distr,varargin)
%GLMFIT Fit a generalized linear model.
%   B = GLMFIT(X,Y,DISTR) fits a generalized linear model using the
%   predictor matrix X, response Y, and distribution DISTR.  The result B
%   is a vector of coefficient estimates.  Acceptable values for DISTR are
%   'normal', 'binomial', 'poisson', 'gamma', and 'inverse gaussian'.  The
%   distribution is fit using the canonical link corresponding to DISTR.
%
%   X is a matrix with rows corresponding to observations, and columns to
%   predictor variables.  GLMFIT automatically includes a constant term in the
%   model (do not enter a column of ones directly into X).  Y is a vector of
%   response values.  If DISTR is 'binomial' Y may a binary vector indicating
%   success/failure, and the total number of trials is taken to be 1 for all
%   observations.  If DISTR is 'binomial', Y may also be a two column matrix,
%   the first column containing the number of successes for each observation,
%   and the second containing the total number of trials.
%
%   GLMFIT treats NaNs in X and Y as missing data, and removes the
%   corresponding observations.
%
%   B = GLMFIT(X,Y,DISTR,'PARAM1',val1,'PARAM2',val2,...) allows you to
%   specify optional parameter name/value pairs to control the model fit.
%   Parameters are:
%
%      'link' - the link function to use in place of the canonical link.
%         The link function defines the relationship f(mu) = x*b
%         between the mean response mu and the linear combination of
%         predictors x*b.  Specify the link parameter value as one of
%            - the text strings 'identity', 'log', 'logit', 'probit',
%              'comploglog', 'reciprocal', 'loglog', or
%            - an exponent P defining the power link, mu = (x*b)^P for
%              x*b > 0, or
%            - a cell array of the form {FL FD FI}, containing three
%              function handles, created using @, that define the link (FL),
%              the derivative of the link (FD), and the inverse link (FI).
%
%      'estdisp' - specify as 'on' to estimate a dispersion parameter for
%         the binomial or Poisson distribution in computing standard
%         errors, or 'off' (the default) to use the theoretical dispersion
%         value.  GLMFIT always estimates the dispersion for other
%         distributions.
%
%      'offset' - a vector to use as an additional predictor variable, but
%         with a coefficient value fixed at 1.0.
%
%      'weights' - a vector of prior weights, such as the inverses of the
%         relative variance of each observation.
%
%      'constant' - specify as 'on' (the default) to include a constant
%         term in the model, or 'off' to omit it.  The coefficient of the
%         constant term is the first element of B.
%
%   [B,DEV] = GLMFIT(...) returns the deviance of the fit.
%
%   [B,DEV,STATS] = GLMFIT(...) returns a structure that contains the
%   following fields:
%       'dfe'       degrees of freedom for error
%       's'         theoretical or estimated dispersion parameter
%       'sfit'      estimated dispersion parameter
%       'se'        standard errors of coefficient estimates B
%       'coeffcorr' correlation matrix for B
%       'covb'      estimated covariance matrix for B
%       't'         t statistics for B
%       'p'         p-values for B
%       'resid'     residuals
%       'residp'    Pearson residuals
%       'residd'    deviance residuals
%       'resida'    Anscombe residuals
%
%   Example:  Fit a probit regression model for y on x.  Each y(i) is the
%   number of successes in n(i) trials.
%
%       x = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
%       n = [48 42 31 34 31 21 23 23 21 16 17 21]';
%       y = [1 2 0 3 8 8 14 17 19 15 17 21]';
%       b = glmfit(x, [y n], 'binomial', 'link', 'probit');
%       yfit = glmval(b, x, 'probit', 'size', n);
%       plot(x, y./n, 'o', x, yfit./n, '-')
%
%   See also GLMVAL, REGSTATS, REGRESS.

%   References:
%      [1] Dobson, A.J. (2002) An Introduction to Generalized Linear
%          Models, 2nd edition, Chapman&Hall/CRC Press.
%      [2] McCullagh, P., and J.A. Nelder (1990) Generalized Linear
%          Models, 2nd edition, Chapman&Hall/CRC Press.
%      [3] Collett, D. (2003) Modelling Binary Data, 2nd edition,
%          Chapman&Hall/CRC Press.

%   The built-in power link is intended for cases where the response data y,
%   and the linear predictor x*b, are positive, and so the link function
%   is calculated as eta = max(mu^p,delta1), where delta1 is a small positive
%   value.  Similarly, the inverse link is mu = max(eta^(1/p),delta2).  It is
%   also possible to define a custom link as
%
%      FL = @(x) sign(x).*abs(x).^p;
%      FD = @(x) p.*abs(x).^(1-p);
%      FI = @(x) sign(x).*abs(x).^(1/p);
%      link = {FL FD FI};
%
%   which may be useful in cases where the data are not positive.

%   Copyright 1993-2015 The MathWorks, Inc.


if nargin < 2
    error(message('stats:glmfit:TooFewInputs'));
end

if nargin < 3 || isempty(distr)
    distr = 'normal';
else
    distr = lower(distr);
end

% Determine the syntax.
if nargin < 4
    newSyntax = false;
else
    arg = varargin{1};
    if ischar(arg) % either a link name (old syntax), or a parameter name
        try
            validatestring(arg, ...
                {'identity' 'log' 'logit' 'probit' 'comploglog' 'reciprocal' 'logloglink'});
            newSyntax = false;
        catch ME
            newSyntax = true;
        end
    else % power link exponent, or custom link, but not a parameter name
        newSyntax = false;
    end
end

% Process optional name/value pairs.
if newSyntax
    paramNames = {     'link' 'estdisp' 'offset' 'weights' 'constant' 'rankwarn'};
    paramDflts = {'canonical'     'off'      []        []        'on'       true};
    [link,estdisp,offset,pwts,const,rankwarn] = ...
                           internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

else % the old syntax glmfit(x,y,distr,link,estdisp,offset,pwts,const)
    link = 'canonical';
    estdisp = 'off';
    offset = [];
    pwts = [];
    const = 'on';
    rankwarn = true;
    if nargin > 3 && ~isempty(varargin{1}), link = varargin{1}; end
    if nargin > 4 && ~isempty(varargin{2}), estdisp = varargin{2}; end
    if nargin > 5 && ~isempty(varargin{3}), offset = varargin{3}; end
    if nargin > 6 && ~isempty(varargin{4}), pwts = varargin{4}; end
    if nargin > 7 && ~isempty(varargin{5}), const = varargin{5}; end
end

estdisp = internal.stats.parseOnOff(estdisp,'''estdisp''');

% Categorical responses 'binomial'
if isa(y,'categorical')
    [y, classname] = grp2idx(y); 
    nc = length(classname);
    if nc > 2
        error(message('stats:glmfit:TwoLevelCategory'));
    end
    y(y==1) = 0;
    y(y==2) = 1;
end

% Set distribution-specific defaults.
N = []; % needed only for binomial
switch distr
case 'normal'
    sqrtvarFun = @(mu) ones(size(mu));
    devFun = @(mu,y) (y - mu).^2;
    if isequal(link, 'canonical'), link  = 'identity'; end
    estdisp = true;
case 'binomial'
    if size(y,2) == 1
        % N will get set to 1 below
        if any(y < 0 | y > 1)
            error(message('stats:glmfit:BadDataBinomialFormat'));
        end
    elseif size(y,2) == 2
        y(y(:,2)==0,2) = NaN;
        N = y(:,2);
        y = y(:,1) ./ N;
        if any(y < 0 | y > 1)
            error(message('stats:glmfit:BadDataBinomialRange'));
        end
    else
        error(message('stats:glmfit:MatrixOrBernoulliRequired'));
    end
    % Wait until N has NaNs removed to define variance function p*(1-p)/N and
    % the deviance function 2*(y*log(y/mu) + (N-y)*log((N-y)/(N-mu))).
    if isequal(link, 'canonical'), link = 'logit'; end
case 'poisson'
    if any(y < 0)
        error(message('stats:glmfit:BadDataPoisson'));
    end
    sqrtvarFun = @(mu) sqrt(mu);
    devFun = @(mu,y) 2*(y .* (log((y+(y==0)) ./ mu)) - (y - mu));
    if isequal(link, 'canonical'), link = 'log'; end
case 'gamma'
    if any(y <= 0)
        error(message('stats:glmfit:BadDataGamma'));
    end
    sqrtvarFun = @(mu) mu;
    devFun = @(mu,y) 2*(-log(y ./ mu) + (y - mu) ./ mu);
    if isequal(link, 'canonical'), link = 'reciprocal'; end
    estdisp = true;
case 'inverse gaussian'
    if any(y <= 0)
        error(message('stats:glmfit:BadDataInvGamma'));
    end
    sqrtvarFun = @(mu) mu.^(3/2);
    devFun = @(mu,y) ((y - mu)./mu).^2 ./  y;
    if isequal(link, 'canonical'), link = -2; end
    estdisp = true;
otherwise
    error(message('stats:glmfit:BadDistribution'));
end

% Remove missing values from the data.  Also turns row vectors into columns.
[anybad,wasnan,y,x,offset,pwts,N] = statremovenan(y,x,offset,pwts,N);
if anybad > 0
    switch anybad
    case 2
        error(message('stats:glmfit:InputSizeMismatchX'))
    case 3
        error(message('stats:glmfit:InputSizeMismatchOffset'))
    case 4
        error(message('stats:glmfit:InputSizeMismatchPWTS'))
%   case 5
        % N is empty, or was created from y (so its length must match)
    end
end

if isequal(const,'on')
    x = [ones(size(x,1),1) x];
end
dataClass = superiorfloat(x,y);
x = cast(x,dataClass);
y = cast(y,dataClass);

% If x is rank deficient (perhaps because it is overparameterized), we will
% warn and remove columns, and the corresponding coefficients and std. errs.
% will be forced to zero.
[n,ncolx] = size(x);
if isempty(pwts)
    [~,R,perm] = qr(x,0);
else
    [~,R,perm] = qr(x .* pwts(:,ones(1,ncolx)),0);
end
if isempty(R)
    rankx = 0;
else
    rankx = sum(abs(diag(R)) > abs(R(1))*max(n,ncolx)*eps(class(R)));
end
if rankx < ncolx
    if rankwarn
        warning(message('stats:glmfit:IllConditioned'));
    end
    perm = perm(1:rankx);
    x = x(:,perm);
else
    perm = 1:ncolx;
end

% Number of observations after removing missing data, number of coeffs after
% removing dependent cols and (possibly) adding a constant term.
[n,p] = size(x);

if isempty(pwts)
    pwts = 1;
elseif any(pwts == 0)
    % A zero weight means ignore the observation, so n is reduced by one.
    % Residuals will be computed, however.
    n = n - sum(pwts == 0);
end
if isempty(offset), offset = 0; end
if isempty(N), N = 1; end

% Define variance and deviance for binomial, now that N has NaNs removed.
if isequal(distr, 'binomial')
    sqrtN = sqrt(N);
    sqrtvarFun = @(mu) sqrt(mu).*sqrt(1-mu) ./ sqrtN;
    devFun = @(mu,y) 2*N.*(y.*log((y+(y==0))./mu) + (1-y).*log((1-y+(y==1))./(1-mu)));
end

% Instantiate functions for one of the canned links, or validate a
% user-defined link specification.
[linkFun,dlinkFun,ilinkFun] = stattestlink(link,dataClass);

% Initialize mu and eta from y.
mu = startingVals(distr,y,N);
eta = linkFun(mu);

% Set up for iterations
iter = 0;
iterLim = 100;
warned = false;
seps = sqrt(eps);
convcrit = 1e-6;
b = zeros(p,1,dataClass);

% Enforce limits on mu to guard against an inverse link that doesn't map into
% the support of the distribution.
switch distr
case 'binomial'
    % mu is a probability, so order one is the natural scale, and eps is a
    % reasonable lower limit on that scale (plus it's symmetric).
    muLims = [eps(dataClass) 1-eps(dataClass)];
case {'poisson' 'gamma' 'inverse gaussian'}
    % Here we don't know the natural scale for mu, so make the lower limit
    % small.  This choice keeps mu^4 from underflowing.  No upper limit.
    muLims = realmin(dataClass).^.25;
end

while iter <= iterLim
    iter = iter+1;

    % Compute adjusted dependent variable for least squares fit
    deta = dlinkFun(mu);
    z = eta + (y - mu) .* deta;

    % Compute IRLS weights the inverse of the variance function
    sqrtirls = abs(deta) .* sqrtvarFun(mu);
    sqrtw = sqrt(pwts) ./ sqrtirls;

    % If the weights have an enormous range, we won't be able to do IRLS very
    % well.  The prior weights may be bad, or the fitted mu's may have too
    % wide a range, which is probably because the data do as well, or because
    % the link function is trying to go outside the distribution's support.
    wtol = max(sqrtw)*eps(dataClass)^(2/3);
    t = (sqrtw < wtol);
    if any(t)
        t = t & (sqrtw ~= 0);
        if any(t)
            sqrtw(t) = wtol;
            if ~warned
                warning(message('stats:glmfit:BadScaling'));
            end
            warned = true;
        end
    end

    % Compute coefficient estimates for this iteration - the IRLS step
    b_old = b;
    [b,R] = wfit(z - offset, x, sqrtw);

    % Form current linear predictor, including offset
    eta = offset + x * b;

    % Compute predicted mean using inverse link function
    mu = ilinkFun(eta);

    % Force mean in bounds, in case the link function is a wacky choice
    switch distr
    case 'binomial'
        if any(mu < muLims(1) | muLims(2) < mu)
        mu = max(min(mu,muLims(2)),muLims(1));
        end
    case {'poisson' 'gamma' 'inverse gaussian'}
        if any(mu < muLims(1))
        mu = max(mu,muLims(1));
        end
    end

    % Check stopping conditions
    if (~any(abs(b-b_old) > convcrit * max(seps, abs(b_old)))), break; end
end
if iter > iterLim
    warning(message('stats:glmfit:IterationLimit'));
end

bb = zeros(ncolx,1,dataClass); bb(perm) = b;

if iter>iterLim && isequal(distr,'binomial')
    diagnoseSeparation(eta,y,N);
end

if nargout > 1
    % Sum components of deviance to get the total deviance.
    di = devFun(mu,y);
    dev = sum(pwts .* di);
end

% Return additional statistics if requested
if nargout > 2
    % Compute the sum of squares used to estimate dispersion, and the
    % Anscombe residuals.
    switch(distr)
    case 'normal'
        ssr = sum(pwts .* (y - mu).^2);
        anscresid = y - mu;
    case 'binomial'
        ssr = sum(pwts .* (y - mu).^2 ./ (mu .* (1 - mu) ./ N));
        t = 2/3;
        anscresid = beta(t,t) * ...
            (betainc(y,t,t)-betainc(mu,t,t)) ./ ((mu.*(1-mu)).^(1/6) ./ sqrt(N));
    case 'poisson'
        ssr = sum(pwts .* (y - mu).^2 ./ mu);
        anscresid = 1.5 * ((y.^(2/3) - mu.^(2/3)) ./ mu.^(1/6));
    case 'gamma'
        ssr = sum(pwts .* ((y - mu) ./ mu).^2);
        anscresid = 3 * (y.^(1/3) - mu.^(1/3)) ./ mu.^(1/3);
    case 'inverse gaussian'
        ssr = sum(pwts .* ((y - mu) ./ mu.^(3/2)).^2);
        anscresid = (log(y) - log(mu)) ./ mu;
    end

    % Compute residuals, using original count scale for binomial
    if (isequal(distr, 'binomial'))
        resid = (y - mu) .* N;
    else
        resid  = y - mu;
    end

    dfe = max(n - p, 0);
    stats.beta = bb;
    stats.dfe = dfe;
    if dfe > 0
        stats.sfit = sqrt(ssr / dfe);
    else
        stats.sfit = NaN;
    end
    if ~estdisp
        stats.s = 1;
        stats.estdisp = false;
    else
        stats.s = stats.sfit;
        stats.estdisp = true;
    end

    % Find coefficient standard errors and correlations
    if ~isnan(stats.s) % dfe > 0 or estdisp == 'off'
        RI = R\eye(p);
        C = RI * RI';
        if estdisp, C = C * stats.s^2; end
        se = sqrt(diag(C)); se = se(:);   % insure vector even if empty
        stats.covb = zeros(ncolx,ncolx,dataClass);
        stats.covb(perm,perm) = C;
        C = C ./ (se * se');
        stats.se = zeros(ncolx,1,dataClass); stats.se(perm) = se;
        stats.coeffcorr = zeros(ncolx,ncolx,dataClass);
        stats.coeffcorr(perm,perm) = C;
        stats.t = NaN(ncolx,1,dataClass); stats.t(perm) = b ./ se;
        if estdisp
            stats.p = 2 * tcdf(-abs(stats.t), dfe);
        else
            stats.p = 2 * normcdf(-abs(stats.t));
        end
    else
        stats.se = NaN(size(bb),class(bb));
        stats.coeffcorr = NaN(length(bb),class(bb));
        stats.t = NaN(size(bb),class(bb));
        stats.p = NaN(size(bb),class(bb));
        stats.covb = NaN(length(bb),class(bb));
    end

    stats.resid  = statinsertnan(wasnan, resid);
    stats.residp = statinsertnan(wasnan, (y - mu) ./ (sqrtvarFun(mu) + (y==mu)));
    stats.residd = statinsertnan(wasnan, sign(y - mu) .* sqrt(max(0,di)));
    stats.resida = statinsertnan(wasnan, anscresid);
    
    stats.wts = 1./sqrtirls.^2;
end


function [b,R] = wfit(y,x,sw)
% Perform a weighted least squares fit
[~,p] = size(x);
yw = y .* sw;
xw = x .* sw(:,ones(1,p));
% No pivoting, no basic solution.  We've removed dependent cols from x, and
% checked the weights, so xw should be full rank.
[Q,R] = qr(xw,0);
b = R \ (Q'*yw);


function mu = startingVals(distr,y,N)
% Find a starting value for the mean, avoiding boundary values
switch distr
case 'poisson'
    mu = y + 0.25;
case 'binomial'
    mu = (N .* y + 0.5) ./ (N + 1);
case {'gamma' 'inverse gaussian'}
    mu = max(y, eps(class(y))); % somewhat arbitrary
otherwise
    mu = y;
end


function diagnoseSeparation(eta,y,N)
% Compute sample proportions, sorted by increasing fitted value
[x,idx] = sort(eta);
if ~isscalar(N)
    N = N(idx);
end
p = y(idx);
if all(p==p(1))   % all sample proportions are the same
    return
end
if x(1)==x(end)   % all fitted probabilities are the same
    return
end

noFront = 0<p(1) && p(1)<1;     % no "front" section as defined below
noEnd = 0<p(end) && p(end)<1;   % no "end" section as defined below
if noFront && noEnd
    % No potential for perfect separation if neither end is perfect
    return
end

% There is at least one observation potentially taking probability 0 or
% 1 at one end or the other with the data sorted by eta. We want to see
% if the data, sorted by eta (x) value, have this form:
%
%            Front                Middle                End
%        ---------------     -----------------     ---------------
%        x(1)<=...<=x(A)  <  x(A+1)=...=x(B-1)  <  x(B)<=...<=x(n)
% with   p(1)=...=p(A)=0                           p(B)=...=p(n)=1
% or     p(1)=...=p(A)=1                           p(B)=...=p(n)=0
%        ---------------     -----------------     ---------------
%        x may vary here     x is constant here    x may vary here
%
% This includes the possibilities:
%     A+1=B  - no middle section
%     A=0    - no perfect fit at the front
%     B=n+1  - no perfect fit at the end
dx = 100*max(eps(x(1)),eps(x(end)));
n = length(p);
if noFront
    A = 0;
else
    A = find(p~=p(1),1,'first')-1;
    cutoff = x(A+1)-dx;
    A = sum(x(1:A)<cutoff);
end

if noEnd
    B = n+1;
else
    B = find(p~=p(end),1,'last')+1;
    cutoff = x(B-1)+dx;
    B = (n+1) - sum(x(B:end)>cutoff);
end

if A+1<B-1
    % There is a middle region with >1 point, see if x varies there
    if x(B-1)-x(A+1)>dx
        return
    end
end

% We have perfect separation that can be defined by some middle point
if A+1==B
    xmid = x(A) + 0.5*(x(B)-x(A));
else
    xmid = x(A+1);
    if isscalar(N)
        pmid = mean(p(A+1:B-1));
    else
        pmid = sum(p(A+1:B-1).*N(A+1:B-1)) / sum(N(A+1:B-1));
    end
end

% Create explanation part for the lower region, if any
if A>=1
    explanation = sprintf('\n   XB<%g: P=%g',xmid,p(1));
else
    explanation = '';
end

% Add explanation part for the middle region, if any
if A+1<B
    explanation = sprintf('%s\n   XB=%g: P=%g',explanation,xmid,pmid);
end
    
% Add explanation part for the upper region, if any
if B<=n
    explanation = sprintf('%s\n   XB>%g: P=%g',explanation,xmid,p(end));
end

warning(message('stats:glmfit:PerfectSeparation', explanation));


