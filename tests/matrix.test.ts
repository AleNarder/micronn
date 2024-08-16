import { Matrix } from "../src/lib/linalg";
import { Vector } from "../src/lib/linalg";

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

test('Matrix dot matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    expect(matrix.dot(m).toArray()).toEqual([[7, 10], [15, 22]]);
});

test('Matrix add scalar', () => {
    expect(matrix.add(1).toArray()).toEqual([[2, 3], [4, 5]]);
});

test('Matrix add matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    expect(matrix.add(m).toArray()).toEqual([[2, 4], [6, 8]]);
});

test('Matrix sub scalar', () => {
    expect(matrix.sub(1).toArray()).toEqual([[0, 1], [2, 3]]);
});

test('Matrix sub matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    expect(matrix.sub(m).toArray()).toEqual([[0, 0], [0, 0]]);
});

test('Matrix mul scalar', () => {
    expect(matrix.mul(2).toArray()).toEqual([[2, 4], [6, 8]]);
});

test('Matrix div scalar', () => {
    expect(matrix.div(2).toArray()).toEqual([[0.5, 1], [1.5, 2]]);
});

test('Matrix mul matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    expect(matrix.mul(m).toArray()).toEqual([[1, 4], [9, 16]]);
});

test('Matrix div matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    expect(matrix.div(m).toArray()).toEqual([[1, 1], [1, 1]]);
});


test('Matrix rand', () => {
    const m = matrix.copy()
    m.rand();
    expect(m.rows_).toBe(2);
    expect(m.cols_).toBe(2);
    expect(m.toArray()).not.toEqual([[1, 2], [3, 4]]);
});

