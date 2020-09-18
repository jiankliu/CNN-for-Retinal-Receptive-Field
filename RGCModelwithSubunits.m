
% A simple model for RGC

function RGCModelwithSubunits

global RefreshRate; %Stimulus refresh rate (Stim frames per second)
global pStim;
global pRF;

RefreshRate = 30;

%set up 'world' parameters
pWorld.dur = 10^5;         % total stimulus duration (seconds)
pWorld.rf = RefreshRate;   % Monitor refresh rate (Hz)
pWorld.dt  = 1/pWorld.rf;
pWorld.n = pWorld.dur/pWorld.dt;

% Linear RF in Time
pRF.maxt = 0.68;  % filter time scale (seconds)
pRF.nt = floor(pRF.maxt/pWorld.dt);

% Define filter shape by Gamma function
pRF.n = [4,5];           % cascades
pRF.theta = [.05,.05];   % sec
pRF.a = [.05,.05];
pRF.delay = 2*pWorld.dt; % delay
RF = make1DOG(pWorld,pRF);
tmSTA = squeeze(RF);

pStim.NumImages = 5000;  % number of stimulus image. 
                         % YOU MAY WANT TO REDUCE THIS NUMBER FOR MEMORY
                         % ISSUE, BUT THERE ARE LESS SPIKES WITH LESS IMAGES
pStim.xPix = 8;          % size of stimulus image
pStim.yPix = 8;          % size of stimulus image
pStim.Nx = pStim.xPix;
pStim.Ny = pStim.yPix;
pStim.imSize = 4;         %size of subunit

% spSTA
spSTA1 = zeros(pStim.xPix, pStim.yPix);
spSTA2 = zeros(pStim.xPix, pStim.yPix);
spSTA3 = zeros(pStim.xPix, pStim.yPix);
spSTA4 = zeros(pStim.xPix, pStim.yPix);
spSTA1(2+[1:pStim.imSize/2], 2+[1:pStim.imSize/2]) = 1;
spSTA2(2+[1:pStim.imSize/2], 2+pStim.imSize/2+[1:pStim.imSize/2]) = 1;
spSTA3(2+pStim.imSize/2+[1:pStim.imSize/2], 2+[1:pStim.imSize/2]) = 1;
spSTA4(2+pStim.imSize/2+[1:pStim.imSize/2], 2+pStim.imSize/2+[1:pStim.imSize/2]) = 1;

% Binary CheckerBoard Stimulus
rng(1000);
CB = rand(pStim.Nx*pStim.Ny*pStim.NumImages,1);
CB = reshape(CB,pStim.Nx*pStim.Ny,pStim.NumImages);
CB(CB>=0.5) = 1;
CB(CB<0.5)  = 0;
CB = CB - 0.5;

Nsub = 4;
sum_RF = zeros(Nsub,pStim.NumImages);
for i=1:Nsub
    if i==1
        spSTA = spSTA1;
    elseif i==2
        spSTA = spSTA2;
    elseif i==3
        spSTA = spSTA3;
    elseif i==4
        spSTA = spSTA4;
    end
    temp = reshape(spSTA,[],1);
    temp = repmat(temp,1,pStim.NumImages);
    stim = CB .* temp;         % conv in space
    stim = sum(stim,1);
    a = sameconv(stim',tmSTA); % conv in time
    a(a<0) = 0;                % nonlinearity
    sum_RF(i,:) = a;           % subunit signal
end

Weight = [1 1 1 1];
output = Weight * sum_RF;

% Output nonlinearity applied here
pModel.a = 2;          %scaling factor
pModel.Vrest = 2.5;    %threshold
r =  output - pModel.Vrest;
r(r<0) = 0;
r = pModel.a*r;        %final firing rate

% Generate Poisson spike response ---------------
np = 1;
rbig = repmat(r*pWorld.dt,np,1); 
spike = ceil(sum(rand(size(rbig))<rbig,1)');  % Output spike train

% now we have input (stimulus images CB) and output spike train (spike)
% One can do various analysis based on the simulated data

% For example: 
% 1: Use CNN to find all subunits 
% 2: Use STNMF to find all subunits

end


function RF = make1DOG(pWorld,pRF)
%
%
%Creates a 3D receptive field (filter) that is a 2D difference of Gaussian (DOG)
%in space and a difference of Gamma functions in time.
%
%Parameters:
%   pWorld.t            Time vector
%   pWorld.n            Number of data points
%
%   pRF.n               1x2 vector contaning # of cascades for the positive
%                       (1st) and negative (2nd) Gamma function
%   pRF.theta           1x2 vector containing time constants for two Gammas
%   pRF.a               1x2 vector containing the amplitudes of two Gammas
%   pRF.nt              Number of time frames for the RF
%

if ~isfield(pRF,'delay')
    pRF.delay = 0;
end

pWorld.t = linspace(0,pWorld.dur,pWorld.n);
t = pWorld.t(1:(min(pRF.nt,length(pWorld.t))))-pRF.delay;

g1 = Gamma(pRF.n(1),pRF.theta(1),t);
g2 = Gamma(pRF.n(2),pRF.theta(2),t);

g = pRF.a(1)*g1 - pRF.a(2)*g2;

g(t<=0) = 0;
g = -g./norm(g);

RF = fliplr(g);

end


function y=Gamma(n,theta,t)
% GAMMA
%	y=Gamma(n,theta,t)
%	returns a gamma function from [0:t];
%	y=(t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1));
%	which is the result of an n stage leaky integrator.
%

flag=0;

if t(1)==0
    t=t(2:length(t));
    flag=1;
end
id=find(t<=0);
t(id)=ones(size(id));
y = (  (theta'*(1./t)).^(1-n).*exp(-(1./(theta'*(1./t)))))./(theta'*ones(size(t))*factorial(n-1));
y(id)=zeros(size(id));
if flag==1
    y=[0;y']';
end

end


function G = sameconv(A, B)
%  G = sameconv(A, B);
%
%  Causally filters singal A with the filter B, giving a column vector with same height as
%  A.  (B not flipped as in standard convolution).
%
%  Note: B should be in the reverse direction.
%
%  Convolution performed efficiently in (zero-padded) Fourier domain.
%


[am, ~] = size(A);
[bm, ~] = size(B);
nn = am+bm-1;

G = ifft(sum(fft(A,nn).*fft(flipud(B),nn),2));
G = G(1:am,:);

end
