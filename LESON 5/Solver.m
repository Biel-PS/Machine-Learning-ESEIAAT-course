classdef Solver < handle

    properties (Access = private)
        tol
        ClassGradient
    end
    
    methods (Access = public)
        function obj = Solver(cParams)
            obj.init(cParams)
        end
        function theta = gradientSolver (obj,tau,initialGuess)
            error = 1;
            theta = initialGuess;
            while (error > obj.tol)
                grad = obj.ClassGradient.solve(theta);
                grad = reshape(grad,[],1);
                theta = theta - tau*grad';
                error = norm(grad);
                fprintf('Error: %d \n',error);
            end    
        end
    end
    methods (Access = private)
        function init (obj,cParams)
            obj.tol = cParams.tol;
            obj.ClassGradient = cParams.ClassGradient;
        end
    end
end