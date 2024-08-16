import { Matrix } from "../../lib/linalg";
import { Invertible } from "../../types";

export const EPSILON = 1e-7;

interface Lossable extends Invertible<Matrix> {
    forward (prediction: Matrix, target: Matrix): number;
    backward (prediction: Matrix, target: Matrix): Matrix;
}

export abstract class Loss implements Lossable {
    abstract forward(prediction: Matrix, target: Matrix): number;
    abstract backward(prediction: Matrix, target: Matrix): Matrix;
}
