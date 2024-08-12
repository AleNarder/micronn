import { Vector } from "../../lib/linalg";

interface Layer {
    forward (input: Vector): Vector;
    backward (outputError: Vector, lr: number): Vector;

    setLabel(label: string): void;
    getLabel(): string;
}