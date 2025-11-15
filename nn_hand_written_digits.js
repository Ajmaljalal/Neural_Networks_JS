import { sigmoid, sigmoidDerivative, randomGaussian, shuffleArray } from './utils.js';

/**
 * Neural Network Class
 * Implements a feedforward neural network with backpropagation learning.
 * @param {Array<number>} sizes - [784, 30, 10]
 */
class NeuralNetwork {
  constructor(sizes) {
    this.numLayers = sizes.length;
    this.sizes = sizes;
    this.weights = []

    // Initialize biases: one vector for each layer except the input layer
    // For sizes [784, 30, 10], we get biases for layers with 30 and 10 neurons
    this.biases = sizes.slice(1).map(numberNeurons =>
      Array.from({ length: numberNeurons }, () => randomGaussian())
    )

    // Initialize weights: matrices connecting each layer to the next
    // For sizes [784, 30, 10]:
    // - weights[0] is a 30x784 matrix (connecting 784 inputs to 30 hidden neurons)
    // - weights[1] is a 10x30 matrix (connecting 30 hidden to 10 output neurons)
    for (let i = 0; i < sizes.length - 1; i++) {
      const rows = sizes[i + 1];    // Number of neurons in next layer
      const cols = sizes[i];        // Number of neurons in current layer

      // Create a matrix of random weights
      const weightMatrix = Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => randomGaussian())
      );
      this.weights.push(weightMatrix);
    }
  }

  /**
   * Feed an input forward through the network to get an output.
   * This is like passing information through all the layers to make a prediction.
   * @param {Array<number>} inputs - The inputs (e.g., pixel values of an image)
   * @returns {Array<number>} - The final output (e.g., probabilities for each digit 0-9)
   */
  feedforward(inputs) {
    // Start with the input layer
    let currentLayerOutputs = inputs;

    // Pass through each layer
    for (let currentLayerIndex = 0; currentLayerIndex < this.weights.length; currentLayerIndex++) {
      const weightsForThisLayer = this.weights[currentLayerIndex];
      const biasesForThisLayer = this.biases[currentLayerIndex];

      // Calculate the next layer outputs which are weighted sum + bias for each neuron in the next layer
      const nextLayerOutputs = [];

      // For each neuron in the NEXT layer
      for (let neuronIndex = 0; neuronIndex < weightsForThisLayer.length; neuronIndex++) {
        const weightsForThisNeuron = weightsForThisLayer[neuronIndex];
        const biasForThisNeuron = biasesForThisLayer[neuronIndex];

        // Start with the bias
        let weightedSum = biasForThisNeuron;

        // Add contribution from each neuron in the PREVIOUS layer
        for (let prevNeuronIndex = 0; prevNeuronIndex < weightsForThisNeuron.length; prevNeuronIndex++) {
          const weight = weightsForThisNeuron[prevNeuronIndex];
          const inputFromPrevNeuron = currentLayerOutputs[prevNeuronIndex];
          weightedSum += weight * inputFromPrevNeuron;
        }

        // Apply activation function and store
        const neuronOutput = sigmoid(weightedSum);
        nextLayerOutputs.push(neuronOutput);
      }

      currentLayerOutputs = nextLayerOutputs;
    }

    return currentLayerOutputs;
  }

  /**
   * 🧠 THE LEARNING ENGINE: Update Mini-Batch using Gradient Descent
   * 
   * This is where the neural network actually LEARNS!
   * 
   * SIMPLE EXPLANATION:
   * Imagine you're teaching a student by showing them examples. After each small batch
   * of examples, you tell them what they got wrong and by how much. They then adjust
   * their thinking (weights and biases) to do better next time.
   * 
   * @param {Array<Object>} miniBatch - A small batch of training examples [{x: [...], y: [...]}, ...]
   * @param {number} learningRate - How big of a step to take when adjusting (eta)
   */
  updateMiniBatch(miniBatch, learningRate) {
    // --------------------------------------------------------------------------------
    // STEP 1: Create "accumulator" buckets to collect all the corrections
    // --------------------------------------------------------------------------------

    // These will store the SUM of all corrections from each example in the batch.
    // Think of them as "suggestion boxes" where each training example drops in
    // their suggestions for how to improve.

    // Initialize: Create empty correction buckets for biases (one array per layer)
    const totalBiasCorrections = this.biases.map(layerBiases =>
      Array(layerBiases.length).fill(0)
    );

    // Initialize: Create empty correction buckets for weights (one matrix per layer)
    const totalWeightCorrections = this.weights.map(layerWeights =>
      Array.from({ length: layerWeights.length }, () =>
        Array(layerWeights[0].length).fill(0)
      )
    );

    // EXAMPLE: For sizes [784, 30, 10]:
    // - totalBiasCorrections[0] = 30 zeros (corrections for hidden layer biases)
    // - totalBiasCorrections[1] = 10 zeros (corrections for output layer biases)
    // - totalWeightCorrections[0] = 30x784 matrix (corrections for first weight layer)
    // - totalWeightCorrections[1] = 10x30 matrix (corrections for second weight layer)

    // --------------------------------------------------------------------------------
    // STEP 2: Process each training example in the mini-batch
    // --------------------------------------------------------------------------------

    for (const trainingExample of miniBatch) {
      // Each 'trainingExample' has:
      // - trainingExample.x: the input (e.g., 784 pixel values)
      // - trainingExample.y: the correct answer (e.g., [0,0,0,1,0,0,0,0,0,0] for digit "3")

      // Run backpropagation to figure out how wrong we were and how to fix it.
      // This returns the corrections for THIS specific example.
      const corrections = this.backprop(trainingExample.x, trainingExample.y);

      // Add these corrections to our accumulator buckets
      // For biases:
      for (let layerIndex = 0; layerIndex < totalBiasCorrections.length; layerIndex++) {
        const biasesInThisLayer = totalBiasCorrections[layerIndex];

        for (let neuronIndex = 0; neuronIndex < biasesInThisLayer.length; neuronIndex++) {
          totalBiasCorrections[layerIndex][neuronIndex] += corrections.biasCorrections[layerIndex][neuronIndex];
        }
      }

      // For weights:
      for (let layerIndex = 0; layerIndex < totalWeightCorrections.length; layerIndex++) {
        const neuronsInThisLayer = totalWeightCorrections[layerIndex];

        for (let neuronIndex = 0; neuronIndex < neuronsInThisLayer.length; neuronIndex++) {
          const weightsForThisNeuron = neuronsInThisLayer[neuronIndex];

          for (let weightIndex = 0; weightIndex < weightsForThisNeuron.length; weightIndex++) {
            totalWeightCorrections[layerIndex][neuronIndex][weightIndex] +=
              corrections.weightCorrections[layerIndex][neuronIndex][weightIndex];
          }
        }
      }
    }

    // --------------------------------------------------------------------------------
    // STEP 3: Update weights and biases using the AVERAGE correction
    // --------------------------------------------------------------------------------

    // Now we have the total corrections from all examples in the batch.
    // We'll adjust the weights and biases by taking a step in the opposite direction
    // (gradient descent = going downhill to reduce error).

    const numberOfExamples = miniBatch.length;
    const adjustmentSize = learningRate / numberOfExamples; // Average out and apply learning rate

    // Update weights: new_weight = old_weight - (adjustmentSize * total_correction)
    for (let layerIndex = 0; layerIndex < this.weights.length; layerIndex++) {
      const neuronsInThisLayer = this.weights[layerIndex];

      for (let neuronIndex = 0; neuronIndex < neuronsInThisLayer.length; neuronIndex++) {
        const weightsForThisNeuron = neuronsInThisLayer[neuronIndex];

        for (let weightIndex = 0; weightIndex < weightsForThisNeuron.length; weightIndex++) {
          this.weights[layerIndex][neuronIndex][weightIndex] -=
            adjustmentSize * totalWeightCorrections[layerIndex][neuronIndex][weightIndex];
        }
      }
    }

    // Update biases: new_bias = old_bias - (adjustmentSize * total_correction)
    for (let layerIndex = 0; layerIndex < this.biases.length; layerIndex++) {
      const biasesInThisLayer = this.biases[layerIndex];

      for (let neuronIndex = 0; neuronIndex < biasesInThisLayer.length; neuronIndex++) {
        this.biases[layerIndex][neuronIndex] -=
          adjustmentSize * totalBiasCorrections[layerIndex][neuronIndex];
      }
    }

    // DONE! The network has learned a tiny bit from this mini-batch.
    // After seeing many mini-batches, it gets better and better at recognizing patterns!
  }

  /**
   * BACKPROPAGATION: Figure out how wrong we were and how to fix it
   * 
   * SIMPLE EXPLANATION:
   * Imagine you're shooting arrows at a target. Backpropagation is like:
   * 1. See where your arrow landed (how wrong you were)
   * 2. Trace back through your entire motion (stance, grip, release)
   * 3. Figure out which parts of your motion to adjust
   * 
   * @param {Array<number>} inputImage - The input (e.g., 784 pixel values)
   * @param {Array<number>} correctAnswer - The correct answer (e.g., [0,0,0,1,0,0,0,0,0,0])
   * @returns {Object} - The corrections needed for biases and weights
   */
  backprop(inputImage, correctAnswer) {
    // Create correction containers (same structure as our biases and weights)
    const biasCorrections = this.biases.map(layerBiases =>
      Array(layerBiases.length).fill(0)
    );
    const weightCorrections = this.weights.map(layerWeights =>
      Array.from({ length: layerWeights.length }, () =>
        Array(layerWeights[0].length).fill(0)
      )
    );

    // --------------------------------------------------------------------------------
    // FORWARD PASS: Send the input through the network and remember everything
    // --------------------------------------------------------------------------------

    let currentLayerOutputs = inputImage; // Start with the input
    const allLayerOutputs = [inputImage]; // Store outputs from each layer
    const allWeightedSums = []; // Store weighted sums (before applying sigmoid)

    // Pass through each layer and remember the values
    for (let layerIndex = 0; layerIndex < this.weights.length; layerIndex++) {
      const weightsForThisLayer = this.weights[layerIndex];
      const biasesForThisLayer = this.biases[layerIndex];

      // Calculate weighted sum for each neuron: z = weights * inputs + bias
      const weightedSumsThisLayer = [];

      for (let neuronIndex = 0; neuronIndex < weightsForThisLayer.length; neuronIndex++) {
        const weightsForThisNeuron = weightsForThisLayer[neuronIndex];
        const biasForThisNeuron = biasesForThisLayer[neuronIndex];

        let weightedSum = biasForThisNeuron;
        for (let prevNeuronIndex = 0; prevNeuronIndex < weightsForThisNeuron.length; prevNeuronIndex++) {
          weightedSum += weightsForThisNeuron[prevNeuronIndex] * currentLayerOutputs[prevNeuronIndex];
        }
        weightedSumsThisLayer.push(weightedSum);
      }
      allWeightedSums.push(weightedSumsThisLayer);

      // Apply sigmoid to get this layer's outputs
      currentLayerOutputs = weightedSumsThisLayer.map(sum => sigmoid(sum));
      allLayerOutputs.push(currentLayerOutputs);
    }

    // EXAMPLE: After forward pass with sizes [784, 30, 10]:
    // - allLayerOutputs[0] = input (784 values)
    // - allLayerOutputs[1] = hidden layer output (30 values)
    // - allLayerOutputs[2] = final output (10 values - our prediction!)

    // --------------------------------------------------------------------------------
    // BACKWARD PASS: Calculate the error and propagate it backwards
    // --------------------------------------------------------------------------------

    // STEP 1: Calculate error at the OUTPUT layer
    // "How wrong were we?"
    const ourPrediction = allLayerOutputs[allLayerOutputs.length - 1];
    const weightedSumsOutputLayer = allWeightedSums[allWeightedSums.length - 1];

    // Calculate error signal for each output neuron
    let errorSignal = [];
    for (let neuronIndex = 0; neuronIndex < ourPrediction.length; neuronIndex++) {
      const whatWePredicted = ourPrediction[neuronIndex];
      const whatItShouldBe = correctAnswer[neuronIndex];
      const howWrongWeWere = whatWePredicted - whatItShouldBe;

      // Multiply by sensitivity (how much this neuron responds to changes)
      const sensitivity = sigmoidDerivative(weightedSumsOutputLayer[neuronIndex]);
      errorSignal.push(howWrongWeWere * sensitivity);
    }

    // Store corrections for the OUTPUT layer biases
    // (The error signal tells us how much to adjust each bias)
    biasCorrections[biasCorrections.length - 1] = errorSignal;

    // Store corrections for the OUTPUT layer weights
    const previousLayerOutputs = allLayerOutputs[allLayerOutputs.length - 2];
    for (let neuronIndex = 0; neuronIndex < errorSignal.length; neuronIndex++) {
      for (let prevNeuronIndex = 0; prevNeuronIndex < previousLayerOutputs.length; prevNeuronIndex++) {
        // Weight correction = error * input that came through this weight
        weightCorrections[weightCorrections.length - 1][neuronIndex][prevNeuronIndex] =
          errorSignal[neuronIndex] * previousLayerOutputs[prevNeuronIndex];
      }
    }

    // STEP 2: Propagate the error BACKWARDS through each hidden layer
    // "Which earlier layers caused this mistake?"
    for (let layerFromEnd = 2; layerFromEnd < this.numLayers; layerFromEnd++) {
      // Get the weighted sums for this layer
      const weightedSumsThisLayer = allWeightedSums[allWeightedSums.length - layerFromEnd];

      // Calculate sensitivity for each neuron in this layer
      const sensitivities = weightedSumsThisLayer.map(sum => sigmoidDerivative(sum));

      // Calculate error signal for this layer by tracing back from next layer
      const errorSignalThisLayer = Array(sensitivities.length).fill(0);
      const weightsFromNextLayer = this.weights[this.weights.length - layerFromEnd + 1];

      // For each neuron in THIS layer
      for (let neuronIndex = 0; neuronIndex < errorSignalThisLayer.length; neuronIndex++) {
        let errorPassedBack = 0;

        // Sum up error from all neurons in the NEXT layer that this neuron connects to
        for (let nextNeuronIndex = 0; nextNeuronIndex < errorSignal.length; nextNeuronIndex++) {
          const connectionWeight = weightsFromNextLayer[nextNeuronIndex][neuronIndex];
          const errorFromNextNeuron = errorSignal[nextNeuronIndex];
          errorPassedBack += connectionWeight * errorFromNextNeuron;
        }

        // Multiply by this neuron's sensitivity
        errorSignalThisLayer[neuronIndex] = errorPassedBack * sensitivities[neuronIndex];
      }

      // Store corrections for this layer's biases
      biasCorrections[biasCorrections.length - layerFromEnd] = errorSignalThisLayer;

      // Store corrections for this layer's weights
      const inputsToThisLayer = allLayerOutputs[allLayerOutputs.length - layerFromEnd - 1];
      for (let neuronIndex = 0; neuronIndex < errorSignalThisLayer.length; neuronIndex++) {
        for (let inputIndex = 0; inputIndex < inputsToThisLayer.length; inputIndex++) {
          weightCorrections[weightCorrections.length - layerFromEnd][neuronIndex][inputIndex] =
            errorSignalThisLayer[neuronIndex] * inputsToThisLayer[inputIndex];
        }
      }

      // Move to the next layer backwards (update error signal)
      errorSignal = errorSignalThisLayer;
    }

    return {
      biasCorrections: biasCorrections,
      weightCorrections: weightCorrections
    };
  }

  /**
   * Evaluate the network's performance on test data.
   * Counts how many predictions are correct.
   * @param {Array} testData - The data to test against.
   * @returns {number} - The number of correct predictions.
   */
  evaluate(testData) {
    let correctCount = 0;

    for (const testExample of testData) {
      // Run the input through the network
      const output = this.feedforward(testExample.x);

      // Find which output neuron has the highest value (our prediction)
      const predictedDigit = output.indexOf(Math.max(...output));

      // Find which output neuron should have been highest (correct answer)
      const actualDigit = testExample.y.indexOf(Math.max(...testExample.y));

      // If they match, we got it right!
      if (predictedDigit === actualDigit) {
        correctCount++;
      }
    }

    return correctCount;
  }

  /**
   * Train the neural network using mini-batch Stochastic Gradient Descent (SGD).
   *
   * @param {Array<Object>} trainingData - All the pictures and correct labels the network learns from.
   * @param {number} epochs - How many times the network should look at ALL the training data.
   * @param {number} miniBatchSize - How many examples the network should look at *before* taking a learning step.
   * @param {number} learningRate - The Learning Rate (how big of a step to take when adjusting).
   * @param {Array<Object>} [testData] - Optional; separate data used to check performance.
   */
  stochasticGradientDescent(trainingData, epochs, miniBatchSize, learningRate, testData = null) {
    // --------------------------------------------------------------------------------
    // 1. SETUP: Finding out how much data we have
    // --------------------------------------------------------------------------------

    // 'n' is the total number of examples the network has to learn from.
    const n = trainingData.length;
    // EXAMPLE: If trainingData has 1000 items, n = 1000.

    // If 'testData' was provided, we count how many test examples we have.
    const nTest = testData ? testData.length : 0;
    // EXAMPLE: If testData has 100 items, nTest = 100.

    // --------------------------------------------------------------------------------
    // 2. THE MAIN LOOP: EPOCHS (The number of times we run the whole lesson)
    // --------------------------------------------------------------------------------

    // We use a standard 'for' loop to run the entire training process 'epochs' times.
    // 'j' is just a counter, starting at 0.
    for (let j = 0; j < epochs; j++) {
      // EXAMPLE: If epochs = 5, this loop runs when j = 0, 1, 2, 3, and 4.

      // --------------------------------------------------------------------------------
      // 3. SHUFFLE: Mix up the cards!
      // --------------------------------------------------------------------------------

      // It's very important to shuffle the data before each epoch.
      // This prevents the network from learning patterns based on the order of the data.
      // (We use a simple utility function for shuffling in JS).
      shuffleArray(trainingData);
      // EXAMPLE: If trainingData was [A, B, C, D], it might now be [C, A, D, B].

      // --------------------------------------------------------------------------------
      // 4. CREATE MINI-BATCHES: Chop the big array into small arrays
      // --------------------------------------------------------------------------------

      const miniBatches = [];
      // We use another 'for' loop to create our small chunks (mini-batches).
      // 'k' starts at 0, and increases by the 'miniBatchSize' each time.
      for (let k = 0; k < n; k += miniBatchSize) {
        // 'k' marks the start of the current chunk.
        // k + miniBatchSize marks the end of the current chunk.

        // .slice(start, end) is like taking a photo of a piece of the original array.
        const miniBatch = trainingData.slice(k, k + miniBatchSize);
        miniBatches.push(miniBatch);

        // EXAMPLE: If n=1000, miniBatchSize=100:
        // k=0: miniBatch = trainingData.slice(0, 100)  -> Contains examples 0-99
        // k=100: miniBatch = trainingData.slice(100, 200) -> Contains examples 100-199
        // ...and so on.
      }
      // EXAMPLE OUTPUT: miniBatches is now [[...100 examples], [...100 examples], ...]

      // --------------------------------------------------------------------------------
      // 5. INNER LOOP: Learning from each small chunk
      // --------------------------------------------------------------------------------

      // Now, we loop over the list of small chunks we just created.
      for (const miniBatch of miniBatches) {
        // For each small chunk, we call the core learning function.
        // This function adjusts the network's internal knowledge (weights/biases).
        this.updateMiniBatch(miniBatch, learningRate);
      }
      // After this loop finishes, the network has successfully 'learned' from all 'n' examples.

      // --------------------------------------------------------------------------------
      // 6. PROGRESS REPORT: Checking how well we did
      // --------------------------------------------------------------------------------

      if (testData) {
        // If we have test data, we check the score!
        const correctCount = this.evaluate(testData);
        // EXAMPLE: If correctCount = 92 and nTest = 100, the output is:
        console.log(`Epoch ${j}: ${correctCount} / ${nTest} correct`);
      } else {
        // If no test data, we just say the full pass is done.
        console.log(`Epoch ${j} complete`);
      }
    } // End of the 'epochs' main loop (j++)
  }
}

// Export the class for use in other modules
export default NeuralNetwork;

