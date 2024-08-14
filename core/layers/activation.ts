import bind from "bind-decorator";
import { Matrix } from "../../lib/linalg";
import { activations } from "../activations";
import { Activation } from "../activations/types";
import { Layer } from "./types";

/**
 * Activation layer
 * This layer applies an activation function to the input
 */
export class ActivationLayer extends Layer{
    
    public readonly activationfn_: Activation;
    public readonly activation_: string;

    private _label!: string;
    private _input!: Matrix;

    constructor(activation: keyof typeof activations) {
        super();
        this.activationfn_ = activations[activation];
        this.activation_   = activation;
    }
    
    @bind
    forward(input: Matrix): Matrix {
        this._input = input;
        const out = input.apply(this.activationfn_.forward);
        return out;
    }
    
    @bind
    backward(outputGradient: Matrix): Matrix {
        const out = this._input.apply(this.activationfn_.backward).mul(outputGradient);
        return out;
    }

    @bind
    setLabel(label: string): void {
        this._label = label;
    }

    @bind
    getLabel(): string {
        return this._label;
    }
}