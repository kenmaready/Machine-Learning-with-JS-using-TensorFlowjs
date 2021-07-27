import * as tf from "@tensorflow/tfjs";



const a = tf.scalar(3);
const b = tf.tensor1d([1, 2, 3, 4]);
const c = tf.tensor2d([1, 2, 3, 4], [2, 2]);
const d = c.sqrt();
d.print();
d.pow(2).print();
d.matMul(d.transpose()).print();
d.mul(d).print();


