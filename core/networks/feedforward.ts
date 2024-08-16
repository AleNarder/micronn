import { Matrix } from "../../lib/linalg";
import { Layer } from "../layers/types";
import { losses } from "../losses";
import { Loss } from "../losses/types";
import { Batch, Network } from "./types";


export class FeedForwardNetwork extends Network {

    readonly layers_: Layer[];
    private _loss: Loss;

    constructor() {
        super();
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
    forward(input: Matrix) {
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
    backward(error: Matrix, lr: number) {
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
    fit(X: Array<Batch>, y: Array<Batch>, lr: number, epochs: number) {

        try {
            
            const Xm = X.map(x => Matrix.fromArray(x));
            const ym = y.map(y => Matrix.fromArray(y));

            for (let epoch = 0; epoch < epochs; epoch++) {

                let error = new Matrix(ym[0].rows_, ym[0].cols_);

                for (let i = 0; i < X.length; i++) {
                    const input  = Xm[i];
                    const target = ym[i];
                    const output = this.forward(input);

                    error = error.add(this._loss.forward(target, output));

                    const err = this._loss.backward(output, target);
                    this.backward(err, lr);
                }

                const hrError = error.div(X.length).flatten().toArray();
                console.log(`Epoch ${epoch + 1}/${epochs}`, 'error', hrError);
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
        const Xm = Xs.map(x => Matrix.fromArray(x));
        return Xm.map(v => this.forward(v));
    }
}