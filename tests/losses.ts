import { losses } from "../src/core/losses";
import { Matrix } from "../src/lib";

test('Mean Squared Error', () => {
    const mse = losses.mse;
    const yTrue = Matrix.fromArray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]);
    
    const yPred = Matrix.fromArray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]);
    
    expect(mse.forward(yTrue, yPred)).toBeCloseTo(0);
    expect(mse.backward(yTrue, yPred).sum()).toBeCloseTo(0);
})

test('Binary Cross Entropy', () => {
    const bce = losses.bce;
    const yTrue = Matrix.fromArray([
        [0],
        [1],
        [1],
        [0]
    ]);
    
    const yPred = Matrix.fromArray([
        [0],
        [1],
        [1],
        [0]
    ]);
    
    expect(bce.forward(yTrue, yPred)).toBeCloseTo(0);
    expect(bce.backward(yTrue, yPred).sum()).toBeCloseTo(0);
})


