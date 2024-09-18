import { bind } from "../../lib";
import { Vector } from "../../lib/linalg";
import { Activation } from "./base";

// Not very useful, but it's here for completeness
export class BinaryStepActivation extends Activation {

    public threshold_: number;

    constructor(threshold: number) {
        super();
        this.threshold_ = threshold;
    }

    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            output.set(i, input.get(i) > this.threshold_ ? 1 : 0);
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            gradient.set(i, 0);
        }
        return gradient;
    }
}

/**
 * Rectified Linear Unit (ReLU) activation function
 * ReLU is a piecewise linear function that will output the input directly if it is positive, 
 * otherwise, it will output zero
 * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
export class ReLuActivation extends Activation {
    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            output.set(i, Math.max(0, input.get(i)));
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            gradient.set(i, input.get(i) > 0 ? 1 : 0);
        }
        return gradient;
    }
}

/**
 * Leaky ReLU activation function
 * Leaky ReLU is an attempt to fix the "dying ReLU" problem by allowing a small,
 * positive gradient when the input is negative
 * 
 * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
export class LeakyReLuActivation extends Activation {

    public alpha_: number;

    constructor(alpha: number) {
        super();
        this.alpha_ = alpha;
    }

    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            output.set(i, Math.max(this.alpha_ * input.get(i), input.get(i)));
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            gradient.set(i, input.get(i) > 0 ? 1 : this.alpha_);
        }
        return gradient;
    }
}


/**
 * Sigmoid activation function, it squashes 
 * the input into the range [0, 1]
 * @see https://en.wikipedia.org/wiki/Sigmoid_function
 */
export class SigmoidActivation extends Activation {
    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            output.set(i, 1 / (1 + Math.exp(-input.get(i))));
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            const sigmoid = 1 / (1 + Math.exp(-input.get(i)));
            gradient.set(i, sigmoid * (1 - sigmoid));
        }
        return gradient;
    }
}

/**
 * Hyperbolic Tangent (tanh) activation function,
 * is a scaled version of the sigmoid function
 * @see https://en.wikipedia.org/wiki/Hyperbolic_function
 */
export class TanhActivation extends Activation {
    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            output.set(i, Math.tanh(input.get(i)));
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            const tanh = Math.tanh(input.get(i));
            gradient.set(i, (1 - Math.pow(tanh, 2)));
        }
        return gradient;
    }
}

/**
 * Softmax function squashes the input into the range [0, 1] 
 * and normalizes the values to sum to 1
 * @see https://en.wikipedia.org/wiki/Softmax_function
 */
export class SoftmaxActivation extends Activation {
    

    forward(input: Vector): Vector {
        const output = new Vector(input.size);

        let sum = 0;
        for (let i = 0; i < input.size; i++) {
            const expValue = Math.exp(input.get(i));
            output.set(i, expValue);
            sum += expValue;
        }
    
        // Normalize
        for (let i = 0; i < input.size; i++) {
            output.set(i, output.get(i) / sum);
        }

        return output;
    }

    /**
     * Softmax backward pass, currently only supports cross entropy loss
     * @param input 
     * @param yTrue 
     * @returns 
     */
    @bind
    backward(input: Vector, yTrue: Vector): Vector {
        const softmax = this.forward(input);
        const gradient = softmax.sub(yTrue);
        return gradient
    }
}

export * from './base';