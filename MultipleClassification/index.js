import * as ml from "./ml.js";
import "./tfjs-vis.js";

const toggleBtn = document.getElementById("toggle-button");
const trainBtn = document.getElementById("train-button");
const testBtn = document.getElementById("test-button");
const loadBtn = document.getElementById("load-button");
const saveBtn = document.getElementById("save-button");
const predictBtn = document.getElementById("predict-button");
const trainStatus = document.getElementById("model-status");
const testStatus = document.getElementById("testing-status");
const predictionInputElement1 = document.getElementById("prediction-input-1");
const predictionInputElement2 = document.getElementById("prediction-input-2");
const predictionOutput = document.getElementById("prediction-output");

toggleBtn.addEventListener("click", toggleVisor);
trainBtn.addEventListener("click", trainModel);
testBtn.addEventListener("click", testModel);
saveBtn.addEventListener("click", saveModel);
loadBtn.addEventListener("click", loadModel);
predictBtn.addEventListener("click", predict);

let data, dp, model;
let modelId = "kc-house-price-multi-classification";

async function toggleVisor() {
    tfvis.visor().toggle();
}

function updateTrainingStatus(message) {
    trainStatus.innerHTML = message;
}

function updateTestingStatus(message) {
    testStatus.innerHTML = message;
}

function updatePredictionOutput(message) {
    predictionOutput.innerHTML = message;
}

function enableButton(button) {
    button.removeAttribute("disabled");
}

function disableButton(button) {
    button.setAttribute("disabled", "disabled");
}

async function trainModel() {
    updateTrainingStatus("Training model...");
    disableButton(trainBtn);
    model = ml.createModel();
    const results = await ml.trainModel(
        model,
        dp,
        dp.trainFeatures,
        dp.trainLabels,
    );
    const trainingLoss = results.history.loss.pop();
    const validationLoss = results.history.val_loss.pop();
    enableButton(testBtn);
    enableButton(trainBtn);
    enableButton(predictBtn);
    updateTrainingStatus(
        `Model trained...\nTraining loss: ${trainingLoss.toPrecision(
            5,
        )}\nValidation loss: ${validationLoss.toPrecision(5)}`,
    );
}

async function testModel() {
    updateTestingStatus("Testing model...");
    const testLoss = await ml.testModel(model, dp.testFeatures, dp.testLabels);
    updateTestingStatus(
        `Testing complete...\nTesting loss: ${testLoss[0].toPrecision(5)}`,
    );
    enableButton(saveBtn);
}

async function saveModel() {
    const saveResults = await ml.saveModel(model, modelId);
    updateTrainingStatus(
        `Trained model saved (${saveResults.modelArtifactsInfo.dateSaved})`,
    );
}

async function loadModel() {
    const loadResults = await ml.loadModel(modelId);
    if (loadResults) {
        model = loadResults;
        tfvis.show.modelSummary({ name: "Model Summary" }, model);
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);

        updateTrainingStatus(
            `Trained model (previously saved model has been loaded)`,
        );
        updateTestingStatus("Current model not tested...");
        enableButton(testBtn);
        enableButton(predictBtn);
    }
}

async function predict() {
    const predictionInput1 = parseInt(predictionInputElement1.value);
    const predictionInput2 = parseInt(predictionInputElement2.value);
    if (isNaN(predictionInput1) || isNaN(predictionInput2)) {
        console.log("No valid number entered. Please enter a valid number.");
    } else if (predictionInput1 < 200 || predictionInput2 < 75000) {
        console.log(
            "Please input a sq foot value of at least 200 and a price of at least $75,000.",
        );
    } else {
        const prediction = await ml.predict(model, dp, [
            predictionInput1,
            predictionInput2,
        ]);
        let predictionString = "";
        prediction.forEach((p, index) => {
            predictionString += ` ${index + 1} br: ${(p * 100).toFixed(2)}% |`;
        });

        updatePredictionOutput(predictionString);
    }
}

async function run() {
    data = ml.getData();
    dp = await ml.process(data);
    updateTrainingStatus("Data loaded. No model trained...");
    enableButton(trainBtn);
    enableButton(loadBtn);
}

run();
