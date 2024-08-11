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

    private input_: Vector;
    private output: Vector;

    constructor(inputSize: number, outputSize: number) {
        this.inputSize_  = inputSize;
        this.outputSize_ = outputSize;
        this.weights_   = new Matrix(outputSize, inputSize);
        this.biases_    = new Vector(outputSize);

        this.weights_.randomize();
        this.biases_.randomize();
    }

    forward(input: Vector) {
        this.input_ = input;
        return this.weights_.dot(input).add(this.biases_);
        
    }
    
    backward(outputGradient: Vector, lr: number ): Vector {
        const inputGradient   = outputGradient.dot(this.weights_.T());
        const weightsGradient = this.input_.T().dot(outputGradient);

        // Update step
        this.weights_ = this.weights_.sub(weightsGradient * lr);
        this.biases_  = this.biases_.sub(outputGradient.mul(lr));

        return inputGradient;
    }
    
    getParameters(): [Matrix, Vector] {
        return [this.weights_, this.biases_];
    }
    
    setParameters(parameters: [Matrix, Vector]): void {
        this.weights_ = parameters[0];
        this.biases_ = parameters[1];
    }
}