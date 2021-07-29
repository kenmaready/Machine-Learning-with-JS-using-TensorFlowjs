export default class DataPreprocessor {
    constructor(dataset, labelName) {
        this.dataset = dataset;
        this.labelName = labelName;
    }

    async initialize() {
        this.data = await this.dataset.toArray();
        if (this.data.length % 2 !== 0) this.data.pop();
    }

    extractFeatures() {
        this.rawFeatures = tf.tensor2d(
            this.data.map((record) => [record.sqft_living, record.price]),
            [this.data.length, 2],
        );
    }

    extractLabels() {
        this.rawLabels = tf.tensor2d(
            this.data.map((record) => record[this.labelName]),
            [this.data.length, 1],
        );
    }

    normalize() {
        this.featureMin = this.rawFeatures.min(0);
        this.featureMax = this.rawFeatures.max(0);
        this.features = this.rawFeatures
            .sub(this.featureMin)
            .div(this.featureMax.sub(this.featureMin));

        this.labelMin = this.rawLabels.min(0);
        this.labelMax = this.rawLabels.max(0);
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

    async scatterplot(predictedPoints = null, equalizeClasses = true) {
        // let data = tf.concat([this.features, this.labels], 1);
        // data = await data.arraySync();

        // let waterfront = data.map((entry) => {
        //     if (entry[2] === 1) return { x: entry[0], y: entry[1] };
        // });
        // let not_waterfront = data.map((entry) => {
        //     if (entry[2] === 0) return { x: entry[0], y: entry[1] };
        // });

        // if (equalizeClasses) {
        //     if (waterfront.length > not_waterfront.length) {
        //         waterfront = waterfront.slice(0, not_waterfront.length);
        //     } else {
        //         not_waterfront = not_waterfront.slice(0, waterfront.length);
        //     }
        // }

        // const values = { not_waterfront, waterfront };
        // const series = ["original"];

        let data = this.data.map((record) => ({
            x: record.sqft_living,
            y: record.price,
            class: record[this.labelName],
        }));

        const allSeries = {};

        data.forEach((p) => {
            const seriesName = `${this.labelName}: ${p.class}`;
            let series = allSeries[seriesName];
            if (!series) {
                series = [];
                allSeries[seriesName] = series;
            }
            series.push(p);
        });

        Object.keys(allSeries).forEach((keyName) => {
            if (allSeries[keyName].length < 100) {
                delete allSeries[keyName];
            }
        });

        // if (Array.isArray(predictedPoints)) {
        //     values.push(predictedPoints);
        //     series.push("predicted");
        // }

        tfvis.render.scatterplot(
            { name: `Sqft_Living vs House Price`, styles: { width: "100%" } },
            {
                values: Object.values(allSeries),
                series: Object.keys(allSeries),
            },
            {
                xLabel: "Sqft_Living",
                yLabel: "Price",
                height: 800,
                width: 1200,
            },
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
