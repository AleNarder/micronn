import { losses} from "../core/losses";
import { Matrix } from "../lib/linalg";

const yTrue = Matrix.fromArray([[0], [1], [1], [0]]);
const yPred = Matrix.fromArray([[0.1], [0.9], [0.8], [0.2]]);

const mse = losses['mse'];

expect(mse.forward(yTrue, yPred).toArray()).toEqual([0.185]);
expect(mse.backward(yTrue, yPred).toArray()).toEqual([[-0.2], [0.4], [0.6], [-0.4]]);

