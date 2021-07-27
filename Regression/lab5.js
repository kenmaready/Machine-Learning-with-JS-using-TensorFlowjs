import * as tf from "@tensorflow/tfjs";

function createModel() {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            inputDim: 1,
            units: 4,
            activation: "linear",
            useBias: true,
        }),
    );
    model.add(
        tf.layers.dense({
            inputDum: 1,
            units: 4,
            activation: "sigmoid",
            useBias: false,
        }),
    );
    model.add(
        tf.layers.dense({
            inputDum: 1,
            units: 2,
            activation: "sigmoid",
            useBias: false,
        }),
    );

    const optimizer = tf.train.sgd(0.1);
    model.compile({ optimizer, loss: "meanSquaredError" });
    return model;
}

const model = createModel();
model.summary();
