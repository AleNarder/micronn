import { Matrix } from "../../lib/linalg";
import { SoftmaxActivation } from "../activations";
import { ActivationLayer, Layer } from "../layers";
import { losses } from "../losses";
import { Loss } from "../losses";
import { Batch, Network } from "./base";
import { writeFileSync } from 'fs';

export class FeedForwardNetwork extends Network {

    readonly layers_: Layer[];
    private _loss: Loss;
    private _losses: number[] = [];
    private _training_time: number = 0;

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

        const lastLayer = this.layers_[this.layers_.length - 1] as ActivationLayer;
        if (lastLayer.activation_ instanceof SoftmaxActivation && loss !== 'crossentropy') {
            throw new Error('Softmax layer requires crossentropy loss');
        }
        
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
        console.log('training with', X.length, 'samples');
        console.log('===========================')
        const start = Date.now();

        try {

            const Xm = X.map(x => Matrix.fromArray(x));
            const ym = y.map(y => Matrix.fromArray(y));

            for (let epoch = 0; epoch < epochs; epoch++) {

                let error = 0

                for (let i = 0; i < X.length; i++) {
                    const input  = Xm[i];
                    const target = ym[i];
                    const output = this.forward(input);

                    error += this._loss.forward(output, target);

                    const err = this._loss.backward(output, target);
                    this.backward(err, lr);
                }

                const hrError = error / X.length;
                console.log(`[${epoch + 1}/${epochs}]:`, hrError);
                this._losses.push(hrError);
            }
            const end = Date.now();
            this._training_time = end - start;

            console.log('===========================')
            console.log("training completed in " + this._training_time + "ms" + "\n");

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
        console.log(Xs)
        const Xm = Xs.map(x => Matrix.fromArray(x));
        return Xm.map(v => this.forward(v));
    }

    /**
     * Calculate the accuracy of the model
     * @param X - the input data
     * @param y - the labels
     */
    accuracy(X: Array<Batch>, y: Array<Batch>, tolerance: number = 0.0) {
        const Xm = X.map(x => Matrix.fromArray(x));
        const ym = y.map(y => Matrix.fromArray(y));

        const report = new Array<{ input: number[][], target: number[][], output: number[][] }>(4);
        
        let correct = 0;
        for (let i = 0; i < X.length; i++) {
            const input  = Xm[i];
            const target = ym[i];
            const output = this.forward(input);

            if (i < report.length) {
                report[i] = ({
                    input: input.toArray(),
                    target: target.toArray(),
                    output: output.toArray()
                })
            }

            if (output.isEqual(target, tolerance)) {
                correct++;
            }
        }

        console.log('===========================')
        console.log(`Accuracy (${tolerance} tol):`, correct / X.length);
        console.log('===========================')
        console.log('Sample report')
        for (const r of report) {
            console.log('predicted:', r.output[0].map(e => parseFloat(e.toFixed(3))), 'true:', r.target[0]);
        }

        return correct / X.length;
    }

    dumpReport() {
        // Create a json file
        const data = JSON.stringify(this._losses);
        writeFileSync('losses.json', data);
    }

}