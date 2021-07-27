// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Generate some synthetic data for training.
let xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
let ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, { epochs: 10 }).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print();
    // Open the browser devtools to see the output
});

const numDisplay = document.getElementById("num-tensors");
numDisplay.innerText = tf.memory().numTensors;

function updateDisplay() {
    numDisplay.innerText = tf.memory().numTensors;
}

// illustrating memory management:
function createLotsofTensors() {
    for (let i = 0; i < 1000; i++) {
        const a = tf.tensor1d([1, 2, 3]);
        const b = tf.scalar(i);
        a.mul(b).print();
        updateDisplay();
    }
}

tf.tidy(createLotsofTensors);

// Lab

const ws = tf.tensor([[1], [2]]);
xs = tf.tensor2d([1, 2, 3, 4], [2, 2]);
ys = xs.mul(tf.scalar(5));
const zs = ys.add(ys).sub(xs);
zs.print();

const as = zs.add(ws);
as.print();

updateDisplay();

function getYs(xs, m, c) {
    return xs.mul(m).add(c);
}

const t1 = tf.tensor1d([1, 5, 10]);
const t2 = getYs(t1, 2, 1);
t2.print();

const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22]);
const max = t3.max();
const min = t3.min();

function normalize(values) {
    const max = values.max();
    const min = values.min();
    return values.sub(min).div(max.sub(min));
}

const t3norm = normalize(t3);
t3norm.print();

const tensors = [t1, t2, t3, max, min, t3norm];
tensors.forEach((t) => tf.dispose(t));

for (let i = 0; i < 100; i++) {
    tf.dispose(tf.tensor([1, 2, 3]));
}

function makeTensors() {
    for (let i = 0; i < 100; i++) {
        tf.tensor([4, 5, 6]);
    }
}

tf.tidy(makeTensors);

updateDisplay();
