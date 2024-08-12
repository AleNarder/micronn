import { losses} from "../core/losses";
import { Vector } from "../lib/linalg";

const yTrue = Vector.fromArray([1, 1, 1]);
const yPred = Vector.fromArray([1, 1, 1]);

/**
 * MEAN SQUARED ERROR
 */

test('Mean Squared Error forward', () => {
    const loss = losses.meanSquaredError;
    expect(loss.forward(yTrue, yPred)).toBeCloseTo(0, 3);
})

test('Mean Squared Error backward', () => {
    const loss = losses.meanSquaredError;
    expect(loss.backward(yTrue, yPred).toArray()).toEqual([0,0,0]);
})
