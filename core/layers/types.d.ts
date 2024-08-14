import { Matrix, Vector } from "../../lib/linalg";

interface Layerable extends Invertible<Matrix> {
    forward (input: Matrix): Matrix;
    backward (outputError: Matrix, lr: number): Matrix;
}

abstract class Layer implements Layerable {
    private label: string;

    abstract forward(input: Matrix): Matrix;
    abstract backward(outputError: Matrix, lr: number): Matrix;

    setLabel(label: string): void {
        this.label = label;
    }

    getLabel(): string {
        return this.label;
    }
}