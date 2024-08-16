// Learn XOR

import { FeedForwardNetwork } from '../core/networks/feedforward';
import { DenseLayer } from '../core/layers/dense';
import { ActivationLayer } from '../core/layers/activation';
import { JSONLoader } from '../lib';
import { TanhActivation } from '../core/activations';
import { resolve } from 'path';

const data = JSONLoader.load<{ value: number[], label: number}>(resolve(__dirname, '..', '..', 'datasets', 'xor.json'))

const trainXs = data.map((x: any) => [x.value])
const trainYs = data.map((x: any) => [[x.label]])


const net = new FeedForwardNetwork();
net.add(new DenseLayer(2, 9))
net.add(new ActivationLayer(
    new TanhActivation()
))
net.add(new DenseLayer(9, 1))
net.add(new ActivationLayer(
    new TanhActivation()
))

net.use('mse');
net.fit(trainXs, trainYs, 0.1, 0.1, 400);
console.log("training completed\n");


net.predict(trainXs).forEach((x, idx) => {
    console.log(`Prediction for ${trainXs[idx]} is ${x.toArray()}`);
});

console.log("\naccuracy:", net.accuracy(trainXs, trainYs, 0.05));