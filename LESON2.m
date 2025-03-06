clear; clc; close all
theta = zeros(10,1);
theta1 =0.7;
theta0 = 0.2;

noiseMag = 1000;

theta = [theta0,theta1]

sampleSize = 100;

x = zeros(sampleSize,2);
x(:,2) = 1:1:sampleSize;
x(:,1) = 1;

y = setData (sampleSize,theta,x,noiseMag);

thetaL2 = normL2(x,y)
fL2 = Functionf(thetaL2,x);

thetaL1 = normL1 (x,y)
fL1 = Functionf(thetaL1,x);

thetaLinf = normLinf (x,y)
fLinf = Functionf(thetaLinf,x);

figure ();
hold on
scatter (x(:,2),y)
plot (x(:,2),fL2,DisplayName='L2')
plot (x(:,2),fL1,DisplayName='L1')
plot (x(:,2),fLinf,DisplayName='Linf')
hold off
legend();

function y = setData (sampleSize,theta,x,noiseMag)
    noiseVec = zeros(sampleSize,1);
    for i = 1:1:sampleSize
        noiseVec(i) = noiseMag*rand;
    end
y = x*theta' + noiseVec;

end

function theta = normL2 (x,y)
    theta = ((x.')*x)\(x.')*y;
end

function theta = normL1 (x,y)
    t = ones (size(x,1),1);
    fun = [0;0;t];
    identity = eye(size(x,1));
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2)

end
function theta = normLinf (x,y)
    fun = [0;0;1];
    identity = ones(size(x,1),1);
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2);

end


function f = Functionf (theta,x)
    f = x*theta;
end
