clear; clc; close all

theta = [0.01,0.02,0.3];
noiseMag = 2;
sampleSize = 10;
dataDeg = 2;
capacity = 6;
trainDataRatio = 0.8;

lam = 0;
computeError = false;

x = generateInitialX(dataDeg,sampleSize);
y = setData (sampleSize,theta,x,noiseMag);

data = splitData (trainDataRatio,x,y);


%%MSE comparison
if (computeError)
    minSampleValue = 10;
    maxSampleValue = 1000;
    errorSampleSize = MSEVarSample (minSampleValue,maxSampleValue,theta,noiseMag,capacity,dataDeg);
    errorCapacityVariation = MSEVarCapacity (data,1,10);
    
    
    figure();
    hold on
    plot (errorSampleSize(:,1),errorSampleSize(:,2),DisplayName='MSE training')
    plot (errorSampleSize(:,1),errorSampleSize(:,3),DisplayName='MSE test');
    title("Fixed capacity, variational sample size");
    xlabel("Number of data points")
    ylabel("MSE")
    hold off
    legend(); grid;
    
    figure();
    hold on
    plot (errorCapacityVariation(:,1),errorCapacityVariation(:,2),DisplayName='MSE training')
    plot (errorCapacityVariation(:,1),errorCapacityVariation(:,3),DisplayName='MSE test');
    title("Fixed sample size, variational capacity");
    xlabel("Capacity (degree of the polynomial)")
    ylabel("MSE")
    hold off
    legend(); grid;

end
    
%%Linear and plynomical model comparison

xLinear = [ones(size(data.train.x)),data.train.x];
yLinear = data.train.y;

thLinear = normL2(xLinear,yLinear);
funLinear = Functionf(thLinear,x(:,1:2));

xTest= setCapacity(capacity,data.test.x);
xTrain= setCapacity(capacity,data.train.x);
yTrain = data.train.y;
xHc = setCapacity(capacity,x(:,2));

thLargeCapacity = normL2(xTrain,yTrain);
funLargeCapacity = Functionf(thLargeCapacity,xHc);


thLam = ThetaLambda (xTrain,yTrain,lam);
funLam = Functionf(thLam,xHc);


figure ();
hold on
scatter (data.train.x,data.train.y,"blue",DisplayName= "training data");
scatter (data.test.x,data.test.y,"red",DisplayName= "test data");

plot (x(:,2),funLinear,DisplayName='Lineal')
plot (x(:,2),funLargeCapacity,DisplayName='Overfitting');
plot(x(:,2), funLam, 'DisplayName', sprintf('Lambda = %.d', lam));

hold off
legend();grid

stepLam = 1;
initialValLam = 0;
finalValLam = 1e3;

[errorLam,error] = MSELamVal (xHc,data,initialValLam,finalValLam,stepLam); %(x,data,initialVal,maxVal,step)
 
errorVec = zeros (size(errorLam,1),2);
errorVec (:,1) = error(1);
errorVec (:,2) = error(2);


figure();
hold on
plot (errorLam(:,1),errorLam(:,2),DisplayName='MSE training for variable \lambda')
plot (errorLam(:,1),errorLam(:,3),DisplayName='MSE test for variable \lambda');
plot (errorLam(:,1),errorVec(:,1),DisplayName='MSE training ')
plot (errorLam(:,1),errorVec(:,2),DisplayName='MSE test ');
title("Fixed capacity and sample size, variable \lambda");
xlabel("\lambda")
ylabel("MSE")
hold off
legend(); grid;


funTest = Functionf(thLargeCapacity,xTest);
funTrain = Functionf(thLargeCapacity,xTrain);

historyGram (data.train.y,funTrain,data.test.y,funTest);

function error = MSEVarSample (initVal,maxVal,thetaInitial,noiseMag,capacity,deg)
    error = zeros ((maxVal-initVal),3); %training and test MSE
    for i = initVal:1:maxVal
        x = zeros(i,3);
        x = generateInitialX(deg,i);
        y = setData (i,thetaInitial,x,noiseMag);
 
        data = splitData (0.8,x,y);
    
        xTrain= setCapacity (capacity,data.train.x);
        yTrain = data.train.y;

        xTest = setCapacity (capacity,data.test.x);
        yTest = data.test.y;
      
        th = normL2(xTrain,yTrain);

        funEvalTrainingCase= Functionf(th,xTrain);
        funEvalTestingCase= Functionf(th,xTest);
            
        MSEtrain = MSE(yTrain,funEvalTrainingCase);
        MSEtest = MSE(yTest,funEvalTestingCase);

        error(i,:) = [i,MSEtrain,MSEtest];
    end  
end

function error = MSEVarCapacity (data,minCapacity,maxCapacity)
    error = zeros ((maxCapacity-minCapacity),3); %training and test MSE
    for i = minCapacity:1:maxCapacity   
        xTrain= setCapacity (i,data.train.x);
        yTrain = data.train.y;

        xTest = setCapacity (i,data.test.x);
        yTest = data.test.y;
      
        thLargeCapacity = normL2(xTrain,yTrain);

        funEvalTrainingCase= Functionf(thLargeCapacity,xTrain);
        funEvalTestingCase= Functionf(thLargeCapacity,xTest);
            
        MSEtrain = MSE(yTrain,funEvalTrainingCase);
        MSEtest = MSE(yTest,funEvalTestingCase);

        error(i,:) = [i,MSEtrain,MSEtest];
    end  
end

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

function f = Functionf (theta,x)
   f = x*theta;
end

function data = splitData (testRatio,x,y)
    sampleSize = size(y,1);
    randomSet = randperm(sampleSize);
    sampleSizeTest = round(sampleSize*testRatio);
    
    randomTest = randomSet(1:sampleSizeTest);
    randomTrain = randomSet((sampleSizeTest+1):end);
    
    data.train.x = x(randomTest,2);
    data.train.y = y(randomTest);
    
    data.test.x = x(randomTrain,2);
    data.test.y = y(randomTrain);
end

function X = setCapacity (n,x) %NO CORSSED TERMS
    prodMatrix = 0:1:(n);
    X = zeros (size(x,1),size(prodMatrix,2));
    if  (1 == size(x,2))
        rhs = x(:,1);
    else
        rhs = x(:,2);
    end
    for i = 1:size(prodMatrix,2)
        X(:,i) = rhs;
    end
    X = X.^prodMatrix;
end

function error = MSE (dataY, fun)
    error = norm(dataY-fun)/size(dataY,1);
end

function error = absError (dataY, fun)
    error =abs(dataY-fun);
end

function x = generateInitialX (deg,sampleSize)
    x = zeros(sampleSize,2);
    x(:,1) = ones(size(x,1),1);
    x(:,2) = (-sampleSize/2):1:(sampleSize/2 -1 );
    
    x = setCapacity(deg,x);
end

function historyGram (dataYTrain,funTrain,dataYTest,funTest)

    errorTrain = absError (dataYTrain, funTrain);
    numPointTrain = 1:1:size(errorTrain,1);

    errorTest = absError (dataYTest, funTest);
    numPointTest = 1:1:size(errorTest,1);

    figure();
    bar(numPointTrain',errorTrain(:));
    xlabel('Point number');
    ylabel('Error MSE in point')
    title('Error per point distribution TRAIN data');
    legend();    
        figure();
    bar(numPointTest', errorTest(:));
    xlabel('Point number');
    ylabel('Error MSE in point')
    title('Error per point distribution TEST data');
    legend();    
end

function theta = ThetaLambda (x,y,lam)
    I = ones(size(x,2),size(x,2));
    theta = ((x.')*x - I.*lam)\(x.')*y;
end

function [errorLam,error] = MSELamVal (x,data,initVal,maxVal,step)

    errorLam = zeros ((maxVal-initVal),3); %training and test MSE
    capacity = size(x,2)-1;

    xTest= setCapacity(capacity,data.test.x);
    xTrain= setCapacity(capacity,data.train.x);
    yTrain = data.train.y;
    yTest = data.test.y;  

    thLargeCapacity = normL2(xTrain,yTrain);
    funTrain = Functionf(thLargeCapacity,xTrain);
    funTest = Functionf(thLargeCapacity,xTest);

    MSEtrain = MSE(yTrain,funTrain);
    MSEtest = MSE(yTest,funTest);

    error = [0,MSEtrain,MSEtest];

    lamSet = initVal:step:maxVal;
    
    for i = 1:1:length(lamSet)   
        th = ThetaLambda (xTrain,yTrain,lamSet(i));   
        funEvalTrainingCase= Functionf(th,xTrain);
        funEvalTestingCase= Functionf(th,xTest);
            
        MSEtrain_lam = MSE(yTrain,funEvalTrainingCase);
        MSEtest_lam = MSE(yTest,funEvalTestingCase);

        errorLam(i,:) = [lamSet(i),MSEtrain_lam,MSEtest_lam];
    end  
end


%% Unused functions

function theta = normL1 (x,y)
    t = ones (size(x,1),1);
    fun = [0;0;t];
    identity = eye(size(x,1));
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2);
end

function theta = normLinf (x,y)
    fun = [0;0;1];
    identity = ones(size(x,1),1);
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2);
end