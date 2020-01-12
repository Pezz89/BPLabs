function [asl_msq, actfact, c0]= asl_P56_Fred_v2 ( x, fs, nbits)
% this implements ITU P.56 method B.
% 'x' is the speech file to calculate active speech level for,
% 'actfact' is the activity factor (between 0 and 1)
%        This is the proportion of the time that the speech is deemed "active"
% 'asl_msq' is the active speech level mean square energy.
%        This is the mean square value in uPa^2 if x is in uPa.
%        For active speech with x in uPa,
%        the Leq in dB re 20 uPa is 10log10[asl_msq/20^2]; 
%        
% 'c0' is the active speech level threshold.
%     thi is the level in uPa above which the speech is deemed active

% Coded by Fred; Commented by BL; 16/6/12.

% x is the column vector of floating point speech data

x = x(:); % make sure x is column vector
T = 0.03; % time constant of smoothing, in seconds
H = 0.2; % hangover time in seconds
M = 15.9; % margin in dB of the difference between threshold and ASL
thres_no = nbits- 1; % number of thresholds, for 16 bit, it's 15

I = ceil( fs* H); % hangover in samples
g = exp( -1/( fs* T)); % smoothing factor in envelop detection
c( 1: thres_no)= 2.^ (-15: thres_no- 16);
% vector with thresholds from one quantizing level up to half the maximum
% code, at a step of 2, in the case of 16bit samples, from 2^-15 to 0.5;

a( 1: thres_no) = 0; % activity counter for each level threshold
hang( 1: thres_no) = I; % hangover counter for each level threshold

sq = x'* x; % long-term level square energy of x
x_len = length( x); % length of x

% use a 2nd order IIR filter to detect the envelope q
x_abs = abs( x);
p = filter( 1-g, [1 -g], x_abs);
q = filter( 1-g, [1 -g], p);  % q is the envelope, obtained from moving average of abs(x) (with slight "hangover").

for k = 1: x_len
    for j = 1: thres_no
        if (q(k)>= c(j))
            a(j) = a(j)+ 1;
            hang(j)= 0;
        elseif (hang(j)< I)
            a(j)= a(j)+ 1;
            hang(j)= hang(j)+ 1;
        else
            break;
        end
    end
end

actfact= 0;
asl_msq= 0;
if (a(1)== 0)
    return;
else
    AdB1= 10* log10( sq/ a(1)+ eps);
end

CdB1= 20* log10( c(1)+ eps);
if (AdB1- CdB1< M)
    return;
end

AdB(1)= AdB1;
CdB(1)= CdB1;
Delta(1)= AdB1- CdB1;

for j= 2: thres_no
    AdB(j)= 10* log10( sq/ (a(j)+ eps)+ eps);
    CdB(j)= 20* log10( c(j)+ eps);
end

for j= 2: thres_no
    if (a(j) ~= 0)
        Delta(j)= AdB(j)- CdB(j);
        if (Delta(j)<= M)
            % interpolate to find the actfact
            [asl_ms_log, cl0]= bin_interp( AdB(j), ...
                AdB(j-1), CdB(j), CdB(j-1), M, 0.5);
            asl_msq= 10^ (asl_ms_log/ 10); % this is the mean square value NOT the rms
            actfact= (sq/ x_len)/ asl_msq; % this is the proportion of the time that the speech is deemed "active"
            c0= 10^( cl0/ 20); % this is the threshold above which the speech is deemed "active".
            break;
        end
    end
end

end

%--------------------------------------------------------------------------

function [asl_ms_log, cc]= bin_interp(upcount, lwcount, ...
    upthr, lwthr, Margin, tol)

if (tol < 0)
    tol = -tol;
end

% Check if extreme counts are not already the true active value
iterno = 1;
if (abs(upcount - upthr - Margin) < tol)
    asl_ms_log= upcount;
    cc= upthr;
    return;
end
if (abs(lwcount - lwthr - Margin) < tol)
    asl_ms_log= lwcount;
    cc= lwthr;
    return;
end

% Initialize first middle for given (initial) bounds
midcount = (upcount + lwcount) / 2.0;
midthr = (upthr + lwthr) / 2.0;

% Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
while ( 1)
    
    diff= midcount- midthr- Margin;
    if (abs(diff)<= tol)
        break;
    end
    
    % if tolerance is not met up to 20 iteractions, then relax the
    % tolerance by 10%
    
    iterno= iterno+ 1;
    
    if (iterno>20)
        tol = tol* 1.1;
    end
    
    if (diff> tol)   % then new bounds are ...
        midcount = (upcount + midcount) / 2.0;
        % upper and middle activities
        midthr = (upthr + midthr) / 2.0;
        % ... and thresholds
    elseif (diff< -tol)	% then new bounds are ...
        midcount = (midcount + lwcount) / 2.0;
        % middle and lower activities
        midthr = (midthr + lwthr) / 2.0;
        % ... and thresholds
    end
    
end
%   Since the tolerance has been satisfied, midcount is selected
%   as the interpolated value with a tol [dB] tolerance.

asl_ms_log= midcount;
cc= midthr;

end


