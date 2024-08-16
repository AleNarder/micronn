# Micronn

A small TypeScript-only Neural Networks library with zero dependencies. It requires Node.js to run, but it can also run in the browser with a few modifications.

## Prerequisites

You need Node.js and npm installed on your computer 

## Features

- [x] FeedForward neural network with momentum and learning rate 
- [x] Dense, Activation and Flatten Layers
- [x] mse, bce, ce loss functions
- [x] linear, step, tanh, sigmoid, softmax, ReLu, leakyReLu activation functions
- [x] ad-hoc linear algebra library

## Run examples

1. Install dependencies 

    ```bash
    npm i
    ```
2. Download example datasets from [here](https://drive.google.com/file/d/1579l2HvQUIAiNrbFpE6hr2iJ_vbVFntJ/view?usp=sharing) and place extracted data in `datasets`

3. Run examples
    
    **XOR**

    ```
    npm run xor
    ```

    **MNIST**
    ```
    npm run mnist
    ```

## Roadmap

- [ ] Convolutional Layers
- [ ] Better logging