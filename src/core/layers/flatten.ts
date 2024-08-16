import { Matrix } from '../../lib/linalg';
import { Layer } from './base';

export class FlattenLayer extends Layer {
    
    private _inputShape!: [number, number];

    forward(input: Matrix) {
        this._inputShape = input.shape();
        return input.flatten();
    }

    backward(input: Matrix) {
        const [ rows, cols ] = this._inputShape;
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                matrix.set(i, j, input.get(0, i * cols + j));
            }
        }
        return matrix;
    }
}