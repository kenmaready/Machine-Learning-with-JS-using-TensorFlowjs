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

toggleBtn.addEventListener("click", toggleVisor);
trainBtn.addEventListener("click", trainModel);
testBtn.addEventListener("click", testModel);
saveBtn.addEventListener("click", saveModel);
loadBtn.addEventListener("click", loadModel);

let data, dp, model;

async function toggleVisor() {
    tfvis.visor().toggle();
}

function updateTrainingStatus(message) {
    trainStatus.innerHTML = message;
}

function updateTestingStatus(message) {
    testStatus.innerHTML = message;
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
    const saveResults = await ml.saveModel(model, "kc-house-price-regression");
    updateTrainingStatus(
        `Trained model saved (${saveResults.modelArtifactsInfo.dateSaved})`,
    );
}

async function loadModel() {}

async function run() {
    data = ml.getData();
    dp = await ml.process(data);
    updateTrainingStatus("Data loaded. No model trained...");
    enableButton(trainBtn);
}

run();
