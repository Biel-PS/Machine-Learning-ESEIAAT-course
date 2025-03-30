classdef neuron

    properties (Access = private)
        numInput
        theta
        x

    end
    
    methods (Access = public)
        function obj = neuron(cParams)
            obj.init(cParams);
        end
        function g = computeNeuron(obj)
            g = obj.theta'*obj.x
        end
    end
    methods (Access = private)
        function obj = init(obj,cParams)
            obj.numInput = cParams.numInput;
            obj.theta = cParams.theta;
            obj.x = cParams.x;
        end
    end
end

