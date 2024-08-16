import { Vector } from "../../lib/linalg"
import { Invertible } from "../../types";

interface Activatable extends Invertible<Vector> {
    forward(input: Vector): Vector;
    backward(input: Vector): Vector;
}

export abstract class Activation implements Activatable {
    abstract forward(input: Vector): Vector;
    abstract backward(input: Vector): Vector;
}
