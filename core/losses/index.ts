
import { Vector } from '../../lib/linalg';

/**
 * Mean Squared Error loss function
 * @see https://en.wikipedia.org/wiki/Mean_squared_error
 */
export class MeanSquaredError {
    forward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .sub(yPred)
            .pow(2)
            .sum() / (yTrue.size);
    }

    backward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .sub(yPred)
            .mul(2 / yTrue.size);
    }
}

/**
 * Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class CrossEntropy {
    forward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .mul(yPred.log())
            .sum() * (-1 / yTrue.size);
    }
    backward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .div(yPred)
            .mul(-1 / yTrue.size);
    }
}

/**
 * Binary Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class BinaryCrossEntropy {
    forward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .mul(yPred.log())
            .add(yTrue
                .sub(1)
                .mul(yPred.sub(1).log())
            )
            .sum() * (-1 / yTrue.size);
    }

    backward(yTrue: Vector, yPred: Vector) {
        return yTrue
            .div(yPred)
            .sub(yTrue
                .sub(1)
                .div(yPred.sub(1))
            )
            .mul(-1 / yTrue.size);
    }
}

/**
 * Softmax Cross Entropy loss function
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
export class SoftmaxCrossEntropy {
    forward(yTrue: Vector, yPred: Vector) {
        let output = yPred.sub(yPred.max()).exp()
        output = output.div(output.sum());
        
        const error = yTrue.mul(output.log());
        return error.sum() * (-1 / yTrue.size);
    }

    backward(yTrue: Vector, yPred: Vector) {
        let output = yPred.sub(yPred.max()).exp();
        output.div(output.sum());

        const gradient = yTrue.div(output).mul(-1 / yTrue.size);
        return gradient;
    }
}

// TODO: lazy load
export const losses = {
    meanSquaredError: new MeanSquaredError(),
    crossEntropy: new CrossEntropy(),
    binaryCrossEntropy: new BinaryCrossEntropy(),
    softmaxCrossEntropy: new SoftmaxCrossEntropy()
};