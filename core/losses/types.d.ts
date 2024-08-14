import { Matrix } from "../../lib/linalg";

interface Lossable extends Invertible<Matrix> {
    forward (prediction: Matrix, target: Matrix): Matrix;
    backward (prediction: Matrix, target: Matrix): Matrix;
}

abstract class Loss implements Lossable {
    abstract forward(prediction: Matrix, target: Matrix): Matrix;
    abstract backward(prediction: Matrix, target: Matrix): Matrix;
}