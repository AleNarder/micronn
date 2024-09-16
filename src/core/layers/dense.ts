import { bind } from '../../lib/decorators/bind';
import { Matrix } from "../../lib/linalg";
import { Layer } from "./base";

/**
 * Dense layer, aka fully connected layer
 */
export class DenseLayer extends Layer {
    
    readonly inputSize_: number;
    readonly outputSize_: number;

    private _weights: Matrix;
    private _biases: Matrix;
    private _input!: Matrix;

    // Velocity terms for momentum
    private _velocityWeights: Matrix;
    private _velocityBiases: Matrix;

    constructor(inputSize: number, outputSize: number) { 

        super();

        this.inputSize_  = inputSize;
        this.outputSize_ = outputSize;

        // Inialize weights terms
        this._weights    = new Matrix(inputSize, outputSize);
        this._velocityWeights = new Matrix(inputSize, outputSize)
        this._weights.rand();

        // Initialize biases terms
        this._biases     = new Matrix(1, outputSize);
        this._velocityBiases  = new Matrix(1, outputSize)
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
}