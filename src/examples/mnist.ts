import { ReLuActivation, SigmoidActivation, SoftmaxActivation, TanhActivation } from '../core/activations';
import { ActivationLayer, DenseLayer } from '../core/layers';
import { FeedForwardNetwork } from '../core/networks';
import { JSONLoader } from '../lib';
import { resolve } from 'path';

// Load the MNIST dataset
// Slice the dataset to reduce the training time
const data = JSONLoader.load<{ image: number[], label: number}>(resolve(__dirname, '..', '..', 'datasets', 'mnist_012.json')).slice(0, 10000)

const SPLIT_SIZE = 0.9
// Normalize the data
const train   = data.slice(0, Math.floor(data.length * SPLIT_SIZE))
const trainXs = train.map((x: any) => [x.image.map((px: number) => px / 255)])
const trainYs = train.map((x: any) => [
    // Perform one-hot encoding
    new Array(3).fill(0).map((_, idx) => Number(x.label == idx))
])

console.log("extracted training data")

const test    = data.slice(Math.floor(data.length * SPLIT_SIZE), Math.floor(data.length))
const testXs  = test.map((x: any) => [x.image.map((px: number) => px / 255)])
const testYs  = test.map((x: any) => [
    // Perform one-hot encoding
    new Array(3).fill(0).map((_, idx) => Number(x.label === idx))
])

console.log("extracted test data")

const net = new FeedForwardNetwork()

net.add(new DenseLayer(28 * 28, 128))
net.add(new ActivationLayer(new ReLuActivation()))
net.add(new DenseLayer(128, 64))
net.add(new ActivationLayer(new ReLuActivation()))
net.add(new DenseLayer(64, 3))
net.add(new ActivationLayer(new SoftmaxActivation()))

net.use('crossentropy')

net.fit(trainXs, trainYs, 0.001, 10000)
console.log("training completed\n")

console.log("accuracy:", net.accuracy(testXs, testYs))