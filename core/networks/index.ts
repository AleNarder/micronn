import { Vector } from "../../lib/linalg";
import { Layer } from "../layers/types";

export class Network {
    
    readonly layers: Layer[];
    readonly lr_: number;

    constructor(lr: number) {
        this.lr_ = lr;
        this.layers = [];
    }

    add(layer: Layer) {
        this.layers.push(layer);
    }

    forward(input: Vector) {
        let output = input;
        for (const layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(input, output) {
        let gradient = output;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(input, this.lr_);
        }
        return gradient;
    }

    fit (X: Vector[], y: Vector[], lr: number, epochs: number) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < X.length; i++) {
                const input = X[i];
                const target = y[i];
                const output = this.forward(input);
                const error = output.sub(target);
                this.backward(input, error);
            }
        }
    }

    predict(X: Vector[]) {
        return X.map(input => this.forward(input));
    }
}