import DataPreprocessor from "./DataPreprocessor.js";

export function getData() {
    return tf.data.csv("http://localhost:3000/kc_house_data.csv");
}

export async function process(data) {
    await tf.ready();
    const dp = new DataPreprocessor(data);
    await dp.initialize();
    dp.shuffle();
    dp.extractFeatures();
    dp.extractLabels();
    dp.normalize();
    dp.split();
    // dp.scatterplot();
    return dp;
}

export function createModel() {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 10,
            useBias: true,
            activation: "linear",
            inputDim: 1,
        }),
    );

    const optimizer = tf.train.sgd(0.1);

    model.compile({ loss: "meanSquaredError", optimizer });
    return model;
}

export async function trainModel(model, dp, XTrain, yTrain) {
    const { onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss"],
    );

    const trainResults = await model.fit(XTrain, yTrain, {
        batchSize: 32,
        epochs: 20,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd,
            onEpochBegin: async function () {
                await dp.plotPredictions(model);
            },
        },
    });

    await dp.plotPredictions(model);
    return trainResults;
}

export async function testModel(model, XTest, yTest) {
    const lossTensor = model.evaluate(XTest, yTest);
    const loss = await lossTensor.dataSync();
    return loss;
}

export async function saveModel(model, storageId) {
    const saveResults = await model.save(`localstorage://${storageId}`);
    return saveResults;
}

export async function loadModel(storageId) {
    const storageKey = `localstorage://${storageId}`;
    const models = await tf.io.listModels();
    const modelInfo = models[storageKey];
    if (modelInfo) {
        const model = await tf.loadLayersModel(storageKey);
        const optimizer = tf.train.sgd(0.1);

        model.compile({ loss: "meanSquaredError", optimizer });
        return model;
    } else {
        console.log("no saved model");
        return undefined;
    }
}

export async function predict(model, dp, input) {
    let prediction;
    await tf.tidy(() => {
        const inputTensor = tf.tensor1d([input]);
        const normalizedTensor = dp.normalizeNewInput(inputTensor);
        const normalizedPredictionTensor = model.predict(normalizedTensor);
        const predictionTensor = dp.denormalizePredictions(
            normalizedPredictionTensor,
        );
        console.log("predictionTensor:", predictionTensor);
        prediction = predictionTensor.dataSync()[0];
        console.log("prediction:", prediction);
    });
    return prediction;
}

function displayModel(model) {}
