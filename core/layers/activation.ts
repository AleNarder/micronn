import bind from "bind-decorator";
import { Vector, Matrix } from "../../lib/linalg";
import { Activation, activations } from "../activations";
import { Layer } from "./types";

/**
 * Activation layer
 * This layer applies an activation function to the input
 */
export class ActivationLayer implements Layer{
    private input_!: Vector;
    private output!: Vector;
    private activation: Activation;
    private label_!: string;

    constructor(activation: keyof typeof activations) {
        this.activation = activations[activation];
    }
    
    @bind
    forward(input: Vector) {
        this.input_ = input;
        this.output = this.activation.forward(input);
        console.log(this.getLabel(), "in", input.toArray(), "out", this.output.toArray());
        return this.output;
    }
    
    @bind
    backward(outputGradient: Vector): Vector {
        return this.activation.backward(this.input_, outputGradient);
    }

    @bind
    setLabel(label: string): void {
        this.label_ = label;
    }

    @bind
    getLabel(): string {
        return this.label_;
    }
}