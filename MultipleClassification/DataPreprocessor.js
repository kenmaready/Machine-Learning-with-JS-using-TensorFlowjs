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
        );
    }

    extractLabels() {
        this.rawLabels = tf.oneHot(
            tf.tensor1d(
                this.data.map((record) =>
                    this._getClassIndex(record[this.labelName]),
                ),
                "int32",
            ),
            5,
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
            class: record[this.labelName] > 4 ? "5+" : record[this.labelName],
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

    async plotPredictions(model, name = "Predicted Class", size = 400) {
        const [valuesPromise, xTicksPromise, yTicksPromise] = tf.tidy(() => {
            const gridSize = 50;
            const predictionColumns = [];
            for (let colIndex = 0; colIndex < gridSize; colIndex++) {
                const colInputs = [];
                const x = colIndex / gridSize;
                for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
                    const y = (gridSize - rowIndex) / gridSize;
                    colInputs.push([x, y]);
                }
                const colPredictions = model.predict(tf.tensor2d(colInputs));
                predictionColumns.push(colPredictions);
            }
            const valuesTensor = tf.stack(predictionColumns);

            const normalizedTicksTensor = tf
                .linspace(0, 1, 50)
                .concat(tf.linspace(0, 1, 50).reverse())
                .reshape([50, 2]);
            const ticksTensor = this.denormalizeFeatures(normalizedTicksTensor);

            const [xTicksTensor, yTicksTensor] = ticksTensor.split(2, 1);

            return [
                valuesTensor.array(),
                xTicksTensor.array(),
                yTicksTensor.array(),
            ];
        });

        const values = await valuesPromise;
        const xTicks = await xTicksPromise;
        const yTicks = await yTicksPromise;

        const xTickLabels = xTicks.map((v) => (v / 1000).toFixed(1) + "k sqft");
        const yTickLabels = yTicks.map(
            (v) => "$" + (v / 1000).toFixed(0) + "k",
        );
        const data = { values, xTickLabels, yTickLabels };

        tfvis.render.heatmap({ name, tab: "Predictions" }, data, {
            height: size,
            domain: [0, 1],
        });

        this.scatterplot();
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

    _getClassIndex(className) {
        let classIndex;
        if (parseInt(className) > 4) classIndex = 4;
        else classIndex = parseInt(className) - 1;
        return classIndex;
    }

    _getClassName(classIndex) {
        if (classIndex >= 4) className = "5+";
        else className = classIndex + 1;
    }
}
