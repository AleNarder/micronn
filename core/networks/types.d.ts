import { Matrix } from "../../lib/linalg";

export type Batch = Array<Array<number>>;

interface Networkable extends Invertible<Matrix> {
    forward (input: Matrix, ...args: any[]): Matrix;
    backward (outputError: Matrix, lr: number, ...args: any[]): Matrix;
    fit (X: Array<Batch>, y: Array<Batch>, lr: number, epochs: number): void;
    predict (input: Array<Batch>): Matrix[];
} 

abstract class Network implements Networkable {
    abstract forward(input: Matrix, ): Matrix;
    abstract backward(outputError: Matrix, lr: number, args: any): Matrix;
    // TODO: fix this
    abstract predict(input: Array<Batch>): Matrix[];
    abstract fit (X: Array<Batch>, y: Array<Batch>, lr: number, epochs: number): void;
}