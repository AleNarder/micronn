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

/**
 * Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class CrossEntropy extends Loss {
    forward(yTrue: Matrix, yPred: Matrix) {
        const err =  yTrue
            .mul(yPred.apply(x => x.log()))
            .sum() * -1 / yTrue.rows_;
    
        return err;
    }

    backward(yTrue: Matrix, yPred: Matrix) {
        // Avoid division by zero
        const yPredClipped = yPred.clip(EPSILON, Number.MAX_VALUE);
        const back =  yTrue
            .div(yPredClipped)
            .mul(-1 / yTrue.rows_);
       
        return back;
    }
}


export * from './base';
// TODO: lazy load
export const losses = {
    mse: new MeanSquaredError(),
    crossentropy: new CrossEntropy(),
};