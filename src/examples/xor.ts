// Learn XOR

import { FeedForwardNetwork } from '../core/networks/feedforward';
import { DenseLayer } from '../core/layers/dense';
import { ActivationLayer } from '../core/layers/activation';
import { JSONLoader } from '../lib';
import { TanhActivation } from '../core/activations';

const data = JSONLoader.load<{ value: number[], label: number}>('../../datasets/xor.json')

const trainXs = data.map((x: any) => [x.value])
const trainYs = data.map((x: any) => [[x.label]])


const net = new FeedForwardNetwork();
net.add(new DenseLayer(2, 7))
net.add(new ActivationLayer(
    new TanhActivation()
))
net.add(new DenseLayer(7, 1))
net.add(new ActivationLayer(
    new TanhActivation()
))

net.use('mse');
net.fit(trainXs, trainYs, 0.1, 0.1, 350);
net.predict(trainXs).forEach((x, idx) => {
    console.log(`Prediction for ${trainXs[idx]} is ${x.toArray()}`);
});

console.log("accuracy:", net.accuracy(trainXs, trainYs));