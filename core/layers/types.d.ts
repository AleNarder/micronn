import { Matrix, Vector } from "../../lib/linalg";

interface Layer {
    forward (input: Matrix): Matrix;
    backward (outputError: Matrix, lr: number): Matrix;

    setLabel(label: string): void;
    getLabel(): string;
}