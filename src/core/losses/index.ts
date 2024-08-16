import { Matrix } from '../../lib/linalg';
import { Loss, EPSILON } from './base';

/**
 * Mean Squared Error loss function
 * @see https://en.wikipedia.org/wiki/Mean_squared_error
 */
export class MeanSquaredError extends Loss {

    forward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .sub(yPred)
            .apply(x => x.pow(2))
            .sum() / yTrue.rows_;
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .sub(yPred)
            .mul(2 / yTrue.rows_);
    }
}

// /**
//  * Cross Entropy loss function
//  * @see https://en.wikipedia.org/wiki/Cross_entropy
//  */
// export class CrossEntropy extends Loss {
//     forward(yTrue: Matrix, yPred: Matrix) {
//         // Avoid log(0)
//         const yPredClipped = yPred.apply(x => x.apply((x) => Math.max(x, EPSILON)));
//         return yTrue
//             .mul(yPredClipped.apply(x => x.log()))
//             .apply(x => x.sum())
//             .mul(-1 / yTrue.rows_);
//     }

//     backward(yTrue: Matrix, yPred: Matrix) {
//         return yTrue
//             .div(yPred)
//             .mul(-1 / yTrue.cols_);
//     }
// }

/**
 * Binary Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class BinaryCrossEntropy extends Loss {
    forward(yTrue: Matrix, yPred: Matrix) {
        // Avoid log(0)
        const yPredClipped = yPred.clip(EPSILON, Number.MAX_VALUE);
        return yTrue.mul(yPredClipped.apply(x => x.log())).add(
            (yTrue.mul(-1).add(1)).mul(yPredClipped.mul(-1).add(1))
        ).sum() / yTrue.rows_ * -1;
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        const yPredClipped = yPred.clip(EPSILON, Number.MAX_VALUE);

        const n = yTrue.div(yPredClipped);
        const d1 = yTrue.mul(-1).add(1)
        const d2 = yPredClipped.mul(-1).add(1).clip(EPSILON, Number.MAX_VALUE);
        const d = d1.div(d2);
        return n.sub(d).mul(-1)
    }
}


export * from './base';
// TODO: lazy load
export const losses = {
    mse: new MeanSquaredError(),
    // crossentropy: new CrossEntropy(),
    bce: new BinaryCrossEntropy(),
};