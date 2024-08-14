import { Matrix, Vector } from "../../lib/linalg"

export interface Activatable extends Invertible<Vector> {
    forward(input: Vector): Vector;
    backward(input: Vector): Vector;
}

abstract class Activation implements Activatable {
    abstract forward(input: Vector): Vector;
    abstract backward(input: Vector): Vector;
}
