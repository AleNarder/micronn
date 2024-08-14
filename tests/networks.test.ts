// Learn XOR

import { Network } from '../core/networks';
import { DenseLayer } from '../core/layers/dense';
import { ActivationLayer } from '../core/layers/activation';

const X = [
    [
        [0, 0]
    ],
    [
        [0, 1]
    ],
    [
        [1, 0]
    ],
    [
        [1, 1]
    ]
];

const y = [
    [
        [0]
    ],
    [
        [1]
    ],
    [
        [1]
    ],
    [
        [0]
    ]
];


const net = new Network();
net.add(new DenseLayer(2, 3))
net.add(new ActivationLayer("tanh"))
net.add(new DenseLayer(3, 1))
net.add(new ActivationLayer("tanh"))

net.use('mse');
net.fit(X, y, 0.1, 1000);
