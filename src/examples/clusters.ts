import { ReLuActivation, SigmoidActivation, SoftmaxActivation, TanhActivation } from '../core/activations';
import { ActivationLayer, DenseLayer } from '../core/layers';
import { FeedForwardNetwork } from '../core/networks';
import { JSONLoader } from '../lib';
import { resolve } from 'path';

// Load the MNIST dataset
// Slice the dataset to reduce the training time
const data = JSONLoader.load<{ data: number[], label: number}>(resolve(__dirname, '..', '..', 'datasets', 'clusters.json'))

const SPLIT_SIZE = 0.9
// Normalize the data
const train   = data.slice(0, Math.floor(data.length * SPLIT_SIZE))
const trainXs = train.map((x: any) => [x.data])
const trainYs = train.map((x: any) => [
    // Perform one-hot encoding
    new Array(3).fill(0).map((_, idx) => Number(x.label == idx))
])

console.log("extracted training data")

const test    = data.slice(Math.floor(data.length * SPLIT_SIZE), Math.floor(data.length))
const testXs  = test.map((x: any) => [x.data])
const testYs  = test.map((x: any) => [
    // Perform one-hot encoding
    new Array(3).fill(0).map((_, idx) => Number(x.label === idx))
])

console.log("extracted test data")

const net = new FeedForwardNetwork()

net.add(new DenseLayer(3, 16))
net.add(new ActivationLayer(new TanhActivation()))
net.add(new DenseLayer(16, 8))
net.add(new ActivationLayer(new TanhActivation()))
net.add(new DenseLayer(8, 3))
net.add(new ActivationLayer(new SoftmaxActivation()))

net.use('mse')

net.fit(trainXs, trainYs, 0.1, 10000)
console.log("training completed\n")

console.log("accuracy:", net.accuracy(testXs, testYs))