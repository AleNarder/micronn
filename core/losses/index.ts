
import { Matrix } from '../../lib/linalg';
import { Loss } from './types';

const EPSILON = 1e-7;

/**
 * Mean Squared Error loss function
 * @see https://en.wikipedia.org/wiki/Mean_squared_error
 */
export class MeanSquaredError extends Loss {

    forward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .sub(yPred)
            .apply(x => x.pow(2).sum().div(x.size))
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .sub(yPred)
            .mul(2 / yTrue.rows_);
    }
}

/**
 * Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class CrossEntropy extends Loss {
    forward(yTrue: Matrix, yPred: Matrix) {
        // Avoid log(0)
        const yPredClipped = yPred.apply(x => x.apply((x) => Math.max(x, EPSILON)));
        return yTrue
            .mul(yPredClipped.apply(x => x.log()))
            .apply(x => x.sum())
            .mul(-1 / yTrue.rows_);
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .div(yPred)
            .mul(-1 / yTrue.cols_);
    }
}

/**
 * Binary Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class BinaryCrossEntropy extends Loss {
    forward(yTrue: Matrix, yPred: Matrix) {
        // Avoid log(0)
        const yPredClipped = yPred.apply(x => x.apply((x) => Math.max(x, EPSILON)));
        return yTrue
            .mul(yPredClipped.apply(x => x.log()))
            .add(yTrue
                .sub(1)
                .mul(yPred.sub(1).apply(x => x.log())
                ))
            .apply(x => x.sum())
            .mul(-1 / yTrue.cols_);
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        return yTrue
            .div(yPred)
            .sub(yTrue
                .sub(1)
                .div(yPred.sub(1))
            )
            .mul(-1 / yTrue.cols_);
    }
}



// TODO: lazy load
export const losses = {
    mse: new MeanSquaredError(),
    crossentropy: new CrossEntropy(),
    binarycrossentropy: new BinaryCrossEntropy(),
};