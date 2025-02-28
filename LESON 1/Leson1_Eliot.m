Ac =  [1,2,5;0,9,5;4,6,12];
Ah =  [1,2,5;0,9,5];
Av =  [1,2;3,4;5,6];
bcv=[8;3;4];
bh=[8;3];
ls=0.001;
tol=1e-9;
clear figure;
rank(Ac);
rank(Ah);
rank(Av);
tiledlayout (3,2);

theta0=[0;0;0];
Thetac=DirectSolver(Ac,bcv);
Thetac_iter = IterativeSolver(Ac,bcv,theta0,ls,tol,"Squared Matrix");

theta0=[0;0;0];
Thetah=DirectSolver(Ah,bh);
Thetah_iter = IterativeSolver(Ah,bh,theta0,ls,tol,"Horizontal Matrix");


theta0=[0;0];
Thetav=DirectSolver(Av,bcv);
Thetav_iter = IterativeSolver(Av,bcv,theta0,ls,tol,"Vertical Matrix");




function Theta = DirectSolver(A,b)  

Theta = zeros(size(A,1),size(A,2));
% if (size(A,1) < size(A,2))
% disp ('Horizontal shape')
% end
Theta = ((A.')*A)\(A.')*b;

end

function th = IterativeSolver(A,b,theta0,ls,tol,titulo)
    th=theta0;
    error = 9999;
    i=1;
    gradplot=[0,0];
    fplot=[0,0];
    while (error>tol)
       
       grad = A'*A*th - A'*b;
       th=th-ls*grad;
       error=norm(grad);
       gradplot(i)=error;
       fplot(i)=0.5*(A*th-b)'*(A*th-b);
       
       i=i+1;
       
       
    end
end

plot(gradplot);
xlabel("number of iterations");
ylabel("Gradient value")
title(titulo);

nexttile;
plot(fplot);
xlabel("number of iterations");
ylabel("function value")

endAc =  [1,2,5;0,9,5;4,6,12];
Ah =  [1,2,5;0,9,5];
Av =  [1,2;3,4;5,6];
bcv=[8;3;4];
bh=[8;3];
ls=0.001;
tol=1e-9;


rank(Ac);
rank(Ah);
rank(Av);

theta0=bcv;
Thetac=DirectSolver(Ac,bcv);
Thetac_iter = IterativeSolver(Ac,bcv,theta0,ls,tol)

theta0=[0;0;0];
Thetah=DirectSolver(Ah,bh);
Thetah_iter = IterativeSolver(Ah,bh,theta0,ls,tol)

theta0=[0;0];
Thetav=DirectSolver(Av,bcv);
Thetav_iter = IterativeSolver(Av,bcv,theta0,ls,tol)




function Theta = DirectSolver(A,b)  
    Theta = zeros(size(A,1),size(A,2));
    % if (size(A,1) < size(A,2))
    % disp ('Horizontal shape')
    % end
    Theta = ((A.')*A)\(A.')*b;
end

function th = IterativeSolver(A,b,theta0,ls,tol)
    th=theta0;
    error = 9999;
    while (error>tol)
       grad = A'*A*th - A'*b;
       th=th-ls*grad;
       error=norm(grad);
       disp (error)
    end
end


