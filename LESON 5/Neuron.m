classdef Neuron < handle

    properties (Access = private)
        numInput
        x

    end
    
    methods (Access = public)
        function obj = Neuron(cParams)
            obj.init(cParams);
        end

        function g = computeNeuron (obj,theta)
            h = obj.computeH(theta);
            g = obj.sigmoid(h);
        end
    end
    methods (Access = private)
        function init(obj,cParams)
            obj.numInput = cParams.numInput;
            obj.x = cParams.x;
        end       
        function h = computeH(obj,theta)
            h = obj.x*theta;
        end
        function sig = sigmoid (obj,h)
            sig = 1./(1+exp(-h'));
        end
        function g= handSide (obj,h)
            g = (h>0)';
        end
    end
end

