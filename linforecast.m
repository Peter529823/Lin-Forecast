function [fy,b,tstat,adjR2,RMSE,MAE,U] = linforecast(y,x,sterrmethod)
%the one-step-ahead linear forecast of y, using x as forecasting variables

%   y <--> var to be forecast (T X 1 vector)
%   x <--> forecasting variables (s) (T X K matrix)
%   sterrmethod <--> optional argument if want to use either Newey-West
%   ('NW') or White ('White')

%   fy <--> forecast
%   b  <--> OLS estimate of loadings
%   tstat -- t-statistic of b
%   adjRsquare -- Rsquare adjusted for number of forecasting vars

Y=y(2:end);
x=x(1:end-1,:);
T=size(x,1);
x=[ones(T,1),x];
K=size(x,2);

invXX=inv(x'*x);
b=invXX*(x'*Y);
fy=x*b;
e=Y-fy;
s2=(e'*e)/(T-K);
varb=s2*invXX;
sterr=sqrt(diag(varb));
R2=1-var(e)/var(Y);
adjR2=1-(T-1)/(T-K)*(1-R2);

if nargin > 2
    if strcmp(sterrmethod,'NW')
        NWlags=0; %to use default value -- Newey-West (1994) plug-in procedure
        if NWlags<=0
            NWlags = floor(4*((T/100)^(2/9)));
        end
        sterr = NeweyWest(e,x,T,K,NWlags);
    elseif strcmp(sterrmethod,'White')
        x_trans = x .* e(:,ones(1,K));
        varb = invXX * (x_trans' * x_trans) * invXX;
        varb = T / (T-K-1) * varb; 
        sterr = sqrt(diag(varb));
    end
end

try tstat=b./sterr; catch, tstat=b./sterr'; end  

RMSE = sqrt(mean(e.^2));
MAE = mean(abs(e));
U = 1-sum(e.^2)/sum((Y-y(1:end-1)).^2); 
    
    function y = NeweyWest(e,X,T,k,L)
        Q = 0;
        for l = 0:L
            w_l = 1-l/(L+1);
            for t = l+1:T
                if (l==0)   % This calculates the S_0 portion
                    Q = Q  + e(t) ^2 * X(t, :)' * X(t,:);
                else        % This calculates the off-diagonal terms
                    Q = Q + w_l * e(t) * e(t-l)* ...
                        (X(t, :)' * X(t-l,:) + X(t-l, :)' * X(t,:));
                end
            end
        end
        Q = (1/(T-k)) .*Q;

        y = sqrt(diag(T.*((X'*X)\Q/(X'*X))));

    end
   
    

end

