import { Vector } from "../../lib/linalg";
import { Layer } from "../layers/types";
import { losses } from "../losses";
import { Loss } from "../losses/types";

export class Network {
    
    readonly layers_: Layer[];
    private _loss: Loss;

    constructor() {
        this.layers_ = [];
        this._loss   = losses['meanSquaredError']; 
    }

    /**
     * Add a layer to the network
     * @param layer : the layer to add
     */
    add(layer: Layer) {
        layer.setLabel(`Layer ${this.layers_.length + 1}`);
        this.layers_.push(layer);
    }

    /**
     * Set the loss function to use
     * @param loss - the loss function to use
     */
    use(loss: keyof typeof losses) {
        this._loss = losses[loss];
    }

    /**
     * Forward pass
     * @param input - the input vector 
     */
    forward(input: Vector) {
        let output = input;
        for (const layer of this.layers_) {
            output = layer.forward(output);
        }
        return output;
    }

    /**
     * Backward pass - backpropagation
     * @param error - the error gradient
     * @param lr -learning rate 
     */
    _backward(error: Vector, lr: number) {
        let gradient = error;
        for (let i = this.layers_.length - 1; i >= 0; i--) {
            gradient = this.layers_[i].backward(error, lr);
        }
        return gradient;
    }

    /**
     * Perform training
     * @param X - the input data
     * @param y - the labels
     * @param lr - learning rate
     * @param epochs - number of epochs
     */
    fit (X: Array<Array<number>>, y: Array<number>, lr: number, epochs: number) {
        
        const X_v = X.map(x => Vector.fromArray(x));
        const y_v = y.map(y => Vector.fromArray([y]));

        for (let epoch = 0; epoch < epochs; epoch++) {
            console.log(`Epoch ${epoch + 1}/${epochs}`);
            let error = 0;
            
            for (let i = 0; i < X.length; i++) {
                const input  = X_v[i];
                const target = y_v[i];
                const output = this.forward(input);
                console.log(`Input: ${input.toArray()} - Output: ${output.toArray()} - Target: ${target.toArray()}`);
                error += this._loss.forward(target, output);

                const err = this._loss.backward(target, output);
                this._backward(err, lr);
            }
        }
    }

    /**
     * Predict the output for a given input
     * @param X - the input data
     */
    predict(Xs: Array<Array<number>>) {
        const X_v = Xs.map(x => Vector.fromArray(x));
        return X_v.map(v => this.forward(v));
    }
}