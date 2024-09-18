import { bind } from "../../lib/decorators/bind";
import { Matrix, Vector } from "../../lib/linalg";
import { Activation } from "../activations/base";
import { Layer } from "./base";

/**
 * Activation layer
 * This layer applies an activation function to the input
 */
export class ActivationLayer extends Layer{
    
    public readonly activation_: Activation;
    private _input!: Matrix;

    constructor(activation: Activation) {
        super();
        this.activation_ = activation
    }
    
    @bind
    forward(input: Matrix): Matrix {

        if (!Matrix.isSafe(input)) {
            throw new Error('Unsafe matrix: \n' + input.toArray() + '\n');
        }

        this._input = input;
        const out = input.apply(this.activation_.forward);
        return out;
    }
    
    @bind
    backward(outputGradient: Matrix): Matrix {

        if (!Matrix.isSafe(outputGradient)) {
            throw new Error('Unsafe matrix' + outputGradient.toArray());
        }

        // TODO: implement with gradient descent
        const outputGradientVec = Vector.fromMatrix(outputGradient);
        const out = this._input.apply((v)=>this.activation_.backward(v, outputGradientVec)).mul(outputGradient);
        return out;
    }

}