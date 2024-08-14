import { Matrix } from "../lib/linalg";
import { Vector } from "../lib/linalg";

const matrix = Matrix.fromArray([[1, 2], [3, 4]]);

test('Matrix sum', () => {
    expect(matrix.sum()).toBe(10);
});

test('Matrix T', () => {
    expect(matrix.T().toArray()).toEqual([[1, 3], [2, 4]]);
});

test('Matrix dot vector', () => {
    expect(matrix.dot(Vector.fromArray([1, 2]))).toEqual(Vector.fromArray([5, 11]));
})
