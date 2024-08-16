import { Matrix } from "../../lib/linalg";
import { Invertible } from "../../types";

export type Batch = Array<Array<number>>;

interface Networkable extends Invertible<Matrix> {
    forward (input: Matrix, ...args: any[]): Matrix;
    backward (outputError: Matrix, lr: number, momentum: number): Matrix;
    fit (X: Array<Batch>, y: Array<Batch>, lr: number, momentum: number, epochs: number): void;
    predict (input: Array<Batch>): Matrix[];
} 

export abstract class Network implements Networkable {
    abstract forward(input: Matrix, ): Matrix;
    abstract backward(outputError: Matrix, lr: number, momentum: number): Matrix;
    // TODO: fix this
    abstract predict(input: Array<Batch>): Matrix[];
    abstract fit (X: Array<Batch>, y: Array<Batch>, lr: number,momentum: number, epochs: number): void;
}