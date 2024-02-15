const assert = require("assert");
const tf = require("@tensorflow/tfjs-node");
const HMM = require("hidden-markov-model-tf"); // JavaScript only (CommonJS)

const [observations, time, states, dimensions] = [5, 7, 3, 2];

// Configure model
const hmm = new HMM({
  states,
  dimensions,
});

async function main() {
  // Set parameters
  // pi: initial probability distribution of states
  // A: transition matrix
  // mu: mean of Gaussian emissions
  // Sigma: covariance of Gaussian emissions
  await hmm.setParameters({
    pi: tf.tensor([0.15, 0.2, 0.65]),
    A: tf.tensor([
      [0.55, 0.15, 0.3],
      [0.45, 0.45, 0.1],
      [0.15, 0.2, 0.65],
    ]),
    mu: tf.tensor([
      [-7.0, -8.0],
      [-1.5, 3.7],
      [-1.7, 1.2],
    ]),
    Sigma: tf.tensor([
      [
        [0.12, -0.01],
        [-0.01, 0.5],
      ],
      [
        [0.21, 0.05],
        [0.05, 0.03],
      ],
      [
        [0.37, 0.35],
        [0.35, 0.44],
      ],
    ]),
  });

  // Define seed
  const seed = 42;

  // Sample data
  const sample = hmm.sample({ observations, time, seed });
  assert.deepEqual(sample.states.shape, [observations, time]);
  assert.deepEqual(sample.emissions.shape, [observations, time, dimensions]);

  // Your data must be a tf.tensor with shape [observations, time, dimensions]
  const data = sample.emissions;

  // Fit model with data
  const results = await hmm.fit(data, {
    seed,
  });
  assert(results.converged);

  // Predict hidden state indices
  const inference = hmm.inference(data);
  assert.deepEqual(inference.shape, [observations, time]);

  // Compute log-likelihood
  const logLikelihood = hmm.logLikelihood(data);
  assert.deepEqual(logLikelihood.shape, [observations]);
  logLikelihood.print();

  // Get parameters
  const { pi, A, mu, Sigma } = hmm.getParameters();
  pi.print();
  A.print();
  mu.print();
  Sigma.print();
}

main().catch(async (e) => {
  console.error(e);
  process.exit(1);
});
