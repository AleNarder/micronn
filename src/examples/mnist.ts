import { ReLuActivation, SoftmaxActivation, TanhActivation } from '../core/activations';
import { ActivationLayer, DenseLayer } from '../core/layers';
import { FeedForwardNetwork } from '../core/networks';
import { JSONLoader } from '../lib';
import { resolve } from 'path';

// Load the MNIST dataset
// Slice the dataset to reduce the training time
const data = JSONLoader.load<{ image: number[], label: number}>(resolve(__dirname, '..', '..', 'datasets', 'mnist.json')).slice(0, 1000)

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
    new Array(10).fill(0).map((_, idx) => Number(x.label === idx))
])

console.log("extracted test data")

const net = new FeedForwardNetwork()

net.add(new DenseLayer(28 * 28, 128))
net.add(new ActivationLayer(new ReLuActivation()))
net.add(new DenseLayer(128, 64))
net.add(new ActivationLayer(new ReLuActivation()))
net.add(new DenseLayer(64, 10))
net.add(new ActivationLayer(new SoftmaxActivation()))

net.use('crossentropy')

net.fit(trainXs, trainYs, 0.1, 100)
console.log("training completed\n")

console.log("accuracy:", net.accuracy(testXs, testYs))