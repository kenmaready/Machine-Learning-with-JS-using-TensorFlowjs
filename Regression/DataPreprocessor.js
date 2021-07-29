export default class DataPreprocessor {
    constructor(dataset) {
        this.dataset = dataset;
    }

    async initialize() {
        this.data = await this.dataset.toArray();
        if (this.data.length % 2 !== 0) this.data.pop();
    }

    extractFeatures() {
        this.rawFeatures = tf.tensor2d(
            this.data.map((record) => record.sqft_living),
            [this.data.length, 1],
        );
    }

    extractLabels() {
        this.rawLabels = tf.tensor2d(
            this.data.map((record) => record.price),
            [this.data.length, 1],
        );
    }

    normalize() {
        this.featureMin = this.rawFeatures.min();
        this.featureMax = this.rawFeatures.max();
        this.features = this.rawFeatures
            .sub(this.featureMin)
            .div(this.featureMax.sub(this.featureMin));

        this.labelMin = this.rawLabels.min();
        this.labelMax = this.rawLabels.max();
        this.labels = this.rawLabels
            .sub(this.labelMin)
            .div(this.labelMax.sub(this.labelMin));
    }

    normalizeNewInput(tensor) {
        tensor = tensor
            .sub(this.featureMin)
            .div(this.featureMax.sub(this.featureMin));
        return tensor;
    }

    denormalizeFeatures(tensor) {
        return tensor
            .mul(this.featureMax.sub(this.featureMin))
            .add(this.featureMin);
    }

    denormalizePredictions(tensor) {
        return tensor.mul(this.labelMax.sub(this.labelMin)).add(this.labelMin);
    }

    async scatterplot(predictedPoints = null) {
        let data = tf.concat([this.rawFeatures, this.rawLabels], 1);
        data = await data.arraySync();
        data = data.map((entry) => {
            return { x: entry[0], y: entry[1] };
        });

        const values = [data];
        const series = ["original"];

        if (Array.isArray(predictedPoints)) {
            values.push(predictedPoints);
            series.push("predicted");
        }

        tfvis.render.scatterplot(
            { name: `Sqft_Living vs House Price` },
            { values, series },
            { xLabel: "Sqft_Living", yLabel: "Price" },
        );
    }

    async plotPredictions(model) {
        const [Xs, ys] = tf.tidy(() => {
            let Xs = tf.linspace(0, 1, 100);
            let ys = model.predict(Xs.reshape([100, 1]));

            // denormalize:
            Xs = this.denormalizeFeatures(Xs);
            ys = this.denormalizePredictions(ys);

            return [Xs.dataSync(), ys.dataSync()];
        });

        const predictionPoints = Array.from(Xs).map((x, index) => {
            return { x, y: ys[index] };
        });

        this.scatterplot(predictionPoints);
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
