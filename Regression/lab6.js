import * as tf from "@tensorflow/tfjs";
import * as tfvis from "./tfjs-vis.js";
import { data, process } from "./index.js";

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

function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    const { onEpochEnd } = tfvis.show.fitCallbacks({ name: "Training Loss" }, [
        "loss",
    ]);
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        epochs: 20,
        shuffle: true,
        callbacks: { onEpochEnd },
    });
}

function testModel(model, testingFeatureTensor, testingLabelTensor) {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    console.log(lossTensor);
    const loss = lossTensor.dataSync();
    console.log(`Testing Loss: ${loss}.`);
}

const dp = process(data);
let model = createModel();
model = trainModel(model, data.trainFeatures, data.trainLabels);
model.summary();
