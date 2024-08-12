import bind from 'bind-decorator';
import { Vector, Matrix } from "../../lib/linalg";
import { Layer } from "./types";

/**
 * Dense layer, aka fully connected layer
 */
export class DenseLayer implements Layer {
    
    readonly inputSize_: number;
    readonly outputSize_: number;

    private weights_: Matrix;
    private biases_: Vector;

    private input_!: Vector;
    private label_!: string;

    constructor(inputSize: number, outputSize: number) {
        this.inputSize_  = inputSize;
        this.outputSize_ = outputSize;
        this.weights_   = new Matrix(outputSize, inputSize);
        this.biases_    = new Vector(outputSize);

        this.weights_.rand();
        this.biases_.rand();
    }

    @bind
    forward(input: Vector) {
        this.input_ = input;
        const output = this.weights_.dot(this.input_ ).add(this.biases_);
        console.log(this.getLabel(), "in", input.toArray(), "out", output.toArray());
        return output;
    }
    
    @bind
    backward(outputGradient: Vector, lr: number ): Vector {
        const inputGradient   = outputGradient.dot(this.weights_.T());
        const weightsGradient = this.input_.T().dot(outputGradient);

        // Update step
        this.weights_ = this.weights_.sub(weightsGradient * lr);
        this.biases_  = this.biases_.sub(outputGradient.mul(lr));

        return inputGradient;
    }
    
    @bind
    getParameters(): [Matrix, Vector] {
        return [this.weights_, this.biases_];
    }
    
    @bind
    setParameters(parameters: [Matrix, Vector]): void {
        this.weights_ = parameters[0];
        this.biases_ = parameters[1];
    }

    @bind
    getLabel(): string {
        return this.label_
    }

    @bind
    setLabel(label: string): void {
        this.label_ = label;
    }
}