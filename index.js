const express = require("express");
const tf = require("@tensorflow/tfjs-node"); // TensorFlow.js for Node.js
const axios = require("axios"); // For fetching the CSV from a URL
const csvParser = require("csv-parser"); // For parsing CSV files
const multer = require("multer"); // Optional if needed for file uploads

const app = express();
app.use(express.json());

// Helper function: Fetch and parse CSV
const fetchAndParseCSV = async (url) => {
  return new Promise((resolve, reject) => {
    const rows = [];
    axios
      .get(url, { responseType: "stream" })
      .then((response) => {
        response.data
          .pipe(csvParser())
          .on("data", (row) => rows.push(row))
          .on("end", () => resolve(rows))
          .on("error", (err) => reject(err));
      })
      .catch(reject);
  });
};

// Helper function: Preprocess CSV data
const preprocessData = (rows) => {
  const features = [];
  const labels = [];

  rows.forEach((row) => {
    const values = Object.values(row).map(Number); // Convert all values to numbers
    features.push(values.slice(0, -1)); // All columns except the last are features
    labels.push(values.slice(-1)); // Last column is the label
  });

  // Convert features to 3D tensor for LSTM: [samples, timesteps, features]
  const timeSteps = 3; // Example time step size
  const lstmData = [];
  for (let i = 0; i <= features.length - timeSteps; i++) {
    lstmData.push(features.slice(i, i + timeSteps));
  }

  return {
    lstmData: tf.tensor3d(lstmData),
    dnnData: tf.tensor2d(features),
    labels: tf.tensor2d(labels),
  };
};

// Helper function: Train LSTM model
const trainLSTM = async (data, labels) => {
  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: 50,
      returnSequences: false,
      inputShape: [data.shape[1], data.shape[2]],
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  await model.fit(data, labels, { epochs: 50, batchSize: 16 });
  return model;
};

// Helper function: Train DNN model
const trainDNN = async (data, labels) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
      inputShape: [data.shape[1]],
    })
  );
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  await model.fit(data, labels, { epochs: 50, batchSize: 16 });
  return model;
};

// Helper function: Calculate Metrics
const calculateMetrics = (predictions, actuals) => {
  const predTensor = tf.tensor1d(predictions);
  const actualTensor = tf.tensor1d(actuals);

  const mse = tf.metrics.meanSquaredError(actualTensor, predTensor).arraySync();
  const mae = tf.metrics
    .meanAbsoluteError(actualTensor, predTensor)
    .arraySync();
  const residuals = actualTensor.sub(predTensor);
  const rss = tf.sum(residuals.square()).arraySync(); // Residual Sum of Squares
  const meanActual = tf.mean(actualTensor).arraySync();
  const tss = tf.sum(actualTensor.sub(meanActual).square()).arraySync(); // Total Sum of Squares
  const r2 = 1 - rss / tss;
  const rmse = Math.sqrt(mse);

  return { mse, mae, r2, rmse };
};

// Endpoint: Train models
app.post("/train", async (req, res) => {
  const { csvUrl } = req.body;

  if (!csvUrl) {
    return res
      .status(400)
      .json({ success: false, error: "CSV URL is required" });
  }

  try {
    // Step 1: Fetch and parse CSV
    const rows = await fetchAndParseCSV(csvUrl);

    // Step 2: Preprocess data
    const { lstmData, dnnData, labels } = preprocessData(rows);

    // Step 3: Train LSTM model
    const lstmModel = await trainLSTM(lstmData, labels);
    const lstmPredictions = lstmModel.predict(lstmData).arraySync().flat();

    // Step 4: Train DNN model
    const dnnModel = await trainDNN(dnnData, labels);
    const dnnPredictions = dnnModel.predict(dnnData).arraySync().flat();

    // Step 5: Calculate metrics for both models
    const actuals = labels.arraySync().flat();
    const lstmMetrics = calculateMetrics(lstmPredictions, actuals);
    const dnnMetrics = calculateMetrics(dnnPredictions, actuals);

    res.json({
      success: true,
      lstmMetrics,
      dnnMetrics,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Start the server
const PORT = 8000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
