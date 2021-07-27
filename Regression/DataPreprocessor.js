export default class DataPreprocessor {
    constructor(dataset) {
        this.dataset = dataset;
    }

    async initialize() {
        this.data = await this.dataset.toArray();
        if (this.data.length % 2 !== 0) this.data.pop();
    }

    extractFeatures() {
        this.features = tf.tensor2d(
            this.data.map((record) => record.sqft_living),
            [this.data.length, 1],
        );
    }

    extractLabels() {
        this.labels = tf.tensor2d(
            this.data.map((record) => record.price),
            [this.data.length, 1],
        );
    }

    normalize() {
        this.featureMin = this.features.min();
        this.featureMax = this.features.max();
        this.features = this.features
            .sub(this.featureMin)
            .div(this.featureMax.sub(this.featureMin));

        this.labelMin = this.labels.min();
        this.labelMax = this.labels.max();
        this.labels = this.labels
            .sub(this.labelMin)
            .div(this.labelMax.sub(this.labelMin));
    }

    denormalize(tensor, min, max) {
        return tensor.mul(max.sub(min)).add(min);
    }

    async scatterplot() {
        let data = tf.concat([this.features, this.labels], 1);
        data = await data.arraySync();
        data = data.map((entry) => {
            return { x: entry[0], y: entry[1] };
        });
        console.log("data:", data);

        tfvis.render.scatterplot(
            { name: `Sqft_Living vs House Price` },
            { values: [data], series: ["original"] },
            { xLabel: "Sqft_Living", yLabel: "Price" },
        );
    }

    shuffle() {
        tf.util.shuffle(this.data);
    }

    split() {
        const [XTrain, XTest] = tf.split(this.features, 2);
        const [yTrain, yTest] = tf.split(this.labels, 2);

        this.trainFeatures = XTrain;
        this.testFeatures = XTest;
        this.trainLabels = yTrain;
        this.testLabels = yTest;
    }
}
