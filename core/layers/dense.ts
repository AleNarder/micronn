import bind from 'bind-decorator';
import { Matrix } from "../../lib/linalg";
import { Layer } from "./types";

/**
 * Dense layer, aka fully connected layer
 */
export class DenseLayer implements Layer {
    
    readonly inputSize_: number;
    readonly outputSize_: number;

    private _weights: Matrix;
    private _biases: Matrix;

    private _input!: Matrix;
    private label_!: string;

    constructor(inputSize: number, outputSize: number) {    
        this.inputSize_  = inputSize;
        this.outputSize_ = outputSize;
        
        this._weights    = new Matrix(inputSize, outputSize);
        this._biases     = new Matrix(1, outputSize);

        this._weights.rand();
        this._biases.rand();
    }

    @bind
    forward(input: Matrix) {
        this._input = input;
        const out = input.dot(this._weights).add(this._biases);
        return out;
    }
    
    @bind
    backward(outputGradient: Matrix, lr: number): Matrix {
        const inputGradient = outputGradient.dot(this._weights.T());
        const weightsError  = this._input.T().dot(outputGradient);
        
        // Update step
        this._weights = this._weights.sub(weightsError.mul(lr));
        this._biases  = this._biases.sub(outputGradient.mul(lr));

        return inputGradient;
    }
    
    @bind
    getParameters(): [Matrix, Matrix] {
        return [this._weights, this._biases];
    }
    
    @bind
    setParameters(parameters: [Matrix, Matrix]): void {
        this._weights = parameters[0];
        this._biases = parameters[1];
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