import { Matrix, Vector } from "../../lib/linalg";
import { Layer } from "../layers/types";
import { losses } from "../losses";
import { Loss } from "../losses/types";

export type Batch = Array<Array<number>>;

export class Network {

    readonly layers_: Layer[];
    private _loss: Loss;

    constructor() {
        this.layers_ = [];
        this._loss = losses['mse'];
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
    _forward(input: Matrix) {
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
    _backward(error: Matrix, lr: number) {
        let gradient = error;
        for (let i = this.layers_.length - 1; i >= 0; i--) {
            gradient = this.layers_[i].backward(gradient, lr);
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
    fit(X: Array<Batch>, y: Array<Array<Array<number>>>, lr: number, epochs: number) {

        try {
            const X_m = X.map(x => Matrix.fromArray(x));
            const y_v = y.map(y => Matrix.fromArray(y));

            for (let epoch = 0; epoch < epochs; epoch++) {

                let error = new Matrix(y_v[0].rows_, y_v[0].cols_);

                for (let i = 0; i < X.length; i++) {
                    const input  = X_m[i];
                    const target = y_v[i];
                    const output = this._forward(input);

                    error = error.add(this._loss.forward(target, output));

                    const err = this._loss.backward(output, target);
                    this._backward(err, lr);
                }

                console.log(`Epoch ${epoch + 1}/${epochs}`, 'error', error.div(X.length).toArray());
            }
        } catch (e) {
            if (e instanceof Error) { 
                console.error(e.message);
                console.error(e.stack)
            } else {
                console.error(e);
            }
        }
    }

    /**
     * Predict the output for a given input
     * @param X - the input data
     */
    predict(Xs: Array<Batch>) {
        const X_m = Xs.map(x => Matrix.fromArray(x));
        return X_m.map(v => this._forward(v));
    }
}