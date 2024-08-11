import { Vector } from "../../lib/linalg";

interface Loss {
    forward (prediction: Vector, target: Vector): number;
    backward (prediction: Vector, target: Vector): Vector;
}