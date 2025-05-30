clear; clc; close all
%Deffine initial parameters
ls=5e-4;
tol=1e-9;
matCase = defMatrices();

% Case n=m (square)
fprintf ('Case n=m\n Direct Solver: \n')
Thetac=DirectSolver(matCase.square.A,matCase.square.b);
fprintf ('Case n=m\n Iterative Solver: \n')
[Thetac_iter,gradVector,ThetaVector] = IterativeSolver(matCase.square.A,matCase.square.b,[0;0;0],ls,tol);

figure();
loglog(ThetaVector(2:end),gradVector(2:end));


% Case n<m (horizontal)
fprintf ('Case n<m\n Direct Solver: \n')
Thetah=DirectSolver(matCase.horizontal.A,matCase.horizontal.b);
fprintf ('Case n<m\n Iterative Solver: \n')
Thetah_iter = IterativeSolver(matCase.horizontal.A,matCase.horizontal.b,[0;5;0],ls,tol);

%Case n>m (vertical)
fprintf ('Case n>m\n Direct Solver: \n')
Thetav=DirectSolver(matCase.vertical.A,matCase.vertical.b);
fprintf ('Case n>m\n Iterative Solver: \n')
Thetav_iter = IterativeSolver(matCase.vertical.A,matCase.vertical.b,[0;0],ls,tol);

%% Functions

function matCase = defMatrices()
    matCase.square.A = [1,2,5;0,9,5;4,6,12];
    matCase.square.b = [8;3;4];
    
    matCase.horizontal.A = [1,2,5;0,9,5];
    matCase.horizontal.b = [8;3];
    
    matCase.vertical.A = [1,2;3,4;5,6];
    matCase.vertical.b = [8;3;4];
    
    % rank(matrixCase.square.A)
    % rank(matrixCase.horizontal.A)
    % rank(matrixCase.vertical.A)
end

function Theta = DirectSolver(A,b)  
    Theta = ((A.')*A)\(A.')*b;
end

function [th,gradVector,ThetaVector] = IterativeSolver(A,b,th0,ls,tol)
    th=th0;
    error = 1;
    i = 1;

    gradVector = zeros(1,1);
    ThetaVector = zeros(1,1);
    
    while (error>tol)
       grad = A'*A*th - A'*b;
       gradVector(end+1) = grad(1);
       ThetaVector(end+1) = th(1);
       th=th-ls*grad;
       error=norm(grad);
       fprintf('Error: %d \n',error)
    end
end