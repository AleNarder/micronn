import { Vector, Matrix } from "../../lib/linalg";
import { Activation, activations } from "../activations";
import { Layer } from "./types";

/**
 * Activation layer
 * This layer applies an activation function to the input
 */
export class ActivationLayer implements Layer{
    private input_: Vector;
    private output: Vector;
    private activation: Activation;
    
    constructor(activation: keyof typeof activations) {
        this.activation = activations[activation];
    }
    
    forward(input: Vector) {
        this.input_ = input;
        this.output = this.activation.forward(input);
        return this.output;
    }
    
    backward(outputGradient: Vector): Vector {
        return this.activation.backward(this.input_, outputGradient);
    }
}