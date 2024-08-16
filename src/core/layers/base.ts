import { Matrix } from "../../lib/linalg";
import { Invertible } from "../../types";

interface Layerable extends Invertible<Matrix> {
    forward (input: Matrix): Matrix;
    backward (outputError: Matrix, lr: number, momentum: number): Matrix;
}

export abstract class Layer implements Layerable {
    private _label!: string;

    abstract forward(input: Matrix): Matrix;
    abstract backward(outputError: Matrix, lr: number, momentum: number): Matrix;

    setLabel(label: string): void {
        this._label = label;
    }

    getLabel(): string {
        return this._label;
    }
}