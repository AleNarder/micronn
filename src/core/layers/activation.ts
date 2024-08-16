import { bind } from "../../lib/decorators/bind";
import { Matrix } from "../../lib/linalg";
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
        this._input = input;
        const out = input.apply(this.activation_.forward);
        return out;
    }
    
    @bind
    backward(outputGradient: Matrix): Matrix {
        const out = this._input.apply(this.activation_.backward).mul(outputGradient);
        return out;
    }

}