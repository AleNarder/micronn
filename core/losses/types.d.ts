import { Matrix } from "../../lib/linalg";

interface Loss {
    forward (prediction: Matrix, target: Matrix): Matrix;
    backward (prediction: Matrix, target: Matrix): Matrix;
}