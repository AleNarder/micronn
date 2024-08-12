import { Vector } from "../lib/linalg";

const vector = Vector.fromArray([1, 2, 3]);

test('Vector sum', () => {
    expect(vector.sum()).toBe(6);
});

test('Vector log', () => {
    expect(vector.log().toArray()).toEqual([0, Math.log(2), Math.log(3)]);
});

test('Vector exp', () => {
    expect(vector.exp().toArray()).toEqual([Math.exp(1), Math.exp(2), Math.exp(3)]);
});

test('Vector pow', () => {
    expect(vector.pow(2).toArray()).toEqual([1, 4, 9]);
});

test('Vector T', () => {
    expect(vector.T().toArray()).toEqual([1, 2, 3]);
});

test('Vector rand', () => {
    vector.rand();
    expect(vector.size).toBe(3);
    expect(vector.toArray()).not.toEqual([1, 2, 3]);
});

test('Vector add scalar', () => {
    expect(vector.add(1).toArray()).toEqual([2, 3, 4]);
});

test('Vector add vector', () => {
    expect(vector.add(Vector.fromArray([1, 2, 3])).toArray()).toEqual([2, 4, 6]);
});

test('Vector sub scalar', () => {
    expect(vector.sub(1).toArray()).toEqual([0, 1, 2]);
});

test('Vector sub vector', () => {
    expect(vector.sub(Vector.fromArray([1, 2, 3])).toArray()).toEqual([0, 0, 0]);
});

test('Vector mul scalar', () => {
    expect(vector.mul(2).toArray()).toEqual([2, 4, 6]);
});

test('Vector div scalar', () => {
    expect(vector.div(2).toArray()).toEqual([0.5, 1, 1.5]);
});

test('Vector dot', () => {
    expect(vector.dot(Vector.fromArray([1, 2, 3]))).toBe(14);
});
