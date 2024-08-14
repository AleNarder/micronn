import { Matrix, Vector } from "../../lib/linalg"

export interface Activation {
    forward(input: Vector): Vector;
    backward(input: Vector): Vector;
}
