import { SoftmaxActivation, TanhActivation } from '../core/activations';
import { ActivationLayer, DenseLayer } from '../core/layers';
import { FeedForwardNetwork } from '../core/networks';
import { JSONLoader } from '../lib';

// Load the MNIST dataset
// Slice the dataset to reduce the training time
const data = JSONLoader.load<{ image: number[], label: number}>('../../datasets/mnist.json').slice(0, 1000)

const SPLIT_SIZE = 0.75
// Normalize the data
const train   = data.slice(0, Math.floor(data.length * SPLIT_SIZE))
const trainXs = train.map((x: any) => [x.image.map((x: number) => x / 255)])
const trainYs = train.map((x: any) => [
    // Perform one-hot encoding
    new Array(10).fill(0).map((_, idx) => x.label == idx ? 1 : 0)
])

console.log("extracted training data")

const test    = data.slice(Math.floor(data.length * SPLIT_SIZE), Math.floor(data.length ))
const testXs  = test.map((x: any) => [x.image.map((x: number) => x / 255)])
const testYs  = test.map((x: any) => [
    // Perform one-hot encoding
    new Array(10).fill(0).map((_, idx) => x.label == idx ? 1 : 0)
])

console.log("extracted test data")

const net = new FeedForwardNetwork()

net.add(new DenseLayer(28 * 28, 100))
net.add(new ActivationLayer(new TanhActivation()))
net.add(new DenseLayer(100, 50))
net.add(new ActivationLayer(new TanhActivation()))
net.add(new DenseLayer(50, 10))
net.add(new ActivationLayer(new SoftmaxActivation()))

net.use('mse')

net.fit(trainXs, trainYs, 0.1, 0.1, 20)


console.log("accuracy:", net.accuracy(testXs, testYs))