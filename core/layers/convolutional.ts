import { Vector } from "../../lib/linalg";
import { Layer } from "./types";

export class Conv2D implements Layer {
 
    forward(input: Vector): Vector {
        throw new Error("Method not implemented.");
    }
 
    backward(outputError: Vector, lr: number): Vector {
        throw new Error("Method not implemented.");
    }
    // TODO: Implement Conv2D layer
}