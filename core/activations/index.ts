import { Vector } from "../../lib/linalg";
import { Activation } from "./types";

/**
 * Linear activation function (identity function)
 * @see https://en.wikipedia.org/wiki/Activation_function
 */
export class Linear extends Activation {
    forward(input: Vector): Vector {
        return input;
    }

    backward(input: Vector): Vector {
        // outputGradient.mul(1)
        return input;
    }
}

/**
 * Rectified Linear Unit (ReLU) activation function
 * ReLU is a piecewise linear function that will output the input directly if it is positive, 
 * otherwise, it will output zero
 * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
export class ReLu extends Activation {
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
            gradient.set(i, input.get(i) > 0 ? input.get(i) : 0);
        }
        return gradient;
    }
}


/**
 * Sigmoid activation function, it squashes 
 * the input into the range [0, 1]
 * @see https://en.wikipedia.org/wiki/Sigmoid_function
 */
export class Sigmoid extends Activation {
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
export class Tanh extends Activation {
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
export class Softmax extends Activation {
    forward(input: Vector): Vector {
        const output = new Vector(input.size);
        let sum = 0;
        for (let i = 0; i < input.size; i++) {
            output.set(i, Math.exp(input.get(i)));
            sum += output.get(i);
        }
        for (let i = 0; i < input.size; i++) {
            output.set(i, output.get(i) / sum);
        }
        return output;
    }

    backward(input: Vector): Vector {
        const gradient = new Vector(input.size);
        for (let i = 0; i < input.size; i++) {
            let sum = 0;
            for (let j = 0; j < input.size; j++) {
                sum += Math.exp(input.get(j));
            }
            let softmax = Math.exp(input.get(i)) / sum;
            gradient.set(i, softmax * (1 - softmax));
        }
        return gradient;
    }
}

// TODO: lazy load
export const activations = {
    linear: new Linear(),
    relu: new ReLu(),
    sigmoid: new Sigmoid(),
    tanh: new Tanh(),
    softmax: new Softmax()
} as const;