// Learn XOR

import { Network } from '../core/networks';
import { DenseLayer } from '../core/layers/dense';
import { ActivationLayer } from '../core/layers/activation';


const network = new Network();
network.add(new DenseLayer(2, 2));
network.add(new ActivationLayer('sigmoid'));
network.add(new DenseLayer(2, 1));

const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [0, 1, 1, 0];

network.use('meanSquaredError');

network.fit(X, y, 0.01, 1);

const output = network.predict(X);

console.log(output);