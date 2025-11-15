import { sigmoid, sigmoidDerivative, randomGaussian, calculateLoss } from './utils.js';

// Configuration
const SIZES = [4, 5, 2];
const LAYER_NAMES = ['Input', 'Hidden', 'Output'];
const LEARNING_RATE = 0.5;

// State
let weights = [];
let biases = [];
let layerValues = [];
let weightedSums = [];
let errors = [];
let weightGradients = [];
let biasGradients = [];

// Snapshots for visualizing before/after updates
let oldWeights = null;
let oldBiases = null;

let currentInput = [0.5, 0.8, 0.3, 0.9];
let currentTarget = [1.0, 0.0];

let phase = 'idle'; // 'idle', 'feedforward', 'backprop', 'update'
let currentLayer = 0;
let currentNeuron = 0;
let currentSubStep = 'idle';
let tempWeightedSum = 0;
let tempWithBias = 0;
let tempOutputError = 0;
let tempSigmoidDeriv = 0;
let tempError = 0;
let tempErrorSum = 0;
let tempErrorContributions = [];
let tempCostValue = 0;
let tempCostDerivative = 0;
let currentWeightIdx = 0;
let autoPlayInterval = null;

function initNetwork() {
  weights = [];
  for (let i = 0; i < SIZES.length - 1; i++) {
    weights.push(
      Array.from({ length: SIZES[i + 1] }, () =>
        Array.from({ length: SIZES[i] }, () => randomGaussian() * 0.5)
      )
    );
  }

  biases = SIZES.slice(1).map(n =>
    Array.from({ length: n }, () => randomGaussian() * 0.5)
  );

  layerValues = SIZES.map((size, idx) => {
    if (idx === 0) return [...currentInput];
    return Array(size).fill(null);
  });

  weightedSums = SIZES.slice(1).map(size => Array(size).fill(null));
  errors = [];
  weightGradients = [];
  biasGradients = [];
  oldWeights = null;
  oldBiases = null;

  phase = 'feedforward';
  currentLayer = 0;
  currentNeuron = 0;
  currentSubStep = 'idle';
}

window.resetNetwork = function () {
  initNetwork();
  updateStatus('✅ Network reset! Ready for feedforward. Click "Next Step" to begin.');
  drawNetwork();
  displayLayerValues();
  displayCalculation();
}

window.nextStep = function () {
  if (phase === 'feedforward') {
    stepFeedforward();
  } else if (phase === 'backprop') {
    stepBackprop();
  } else if (phase === 'update') {
    stepUpdate();
  } else if (phase === 'complete') {
    updateStatus('🎉 Training cycle complete! Reset to start again.');
  }
}

function stepFeedforward() {
  if (currentLayer >= SIZES.length - 1) {
    phase = 'backprop';
    currentLayer = SIZES.length - 1;
    currentNeuron = 0;
    currentSubStep = 'start_error';
    updateStatus('✅ Feedforward complete! Starting backpropagation...');
    drawNetwork();
    displayLayerValues();
    displayCalculation();
    return;
  }

  if (currentSubStep === 'idle') {
    currentSubStep = 'weighted_sum';
    tempWeightedSum = 0;

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      tempWeightedSum += weights[currentLayer][currentNeuron][i] * layerValues[currentLayer][i];
    }

  } else if (currentSubStep === 'weighted_sum') {
    currentSubStep = 'add_bias';
    tempWithBias = tempWeightedSum + biases[currentLayer][currentNeuron];

  } else if (currentSubStep === 'add_bias') {
    currentSubStep = 'sigmoid';
    const result = sigmoid(tempWithBias);
    layerValues[currentLayer + 1][currentNeuron] = result;
    weightedSums[currentLayer][currentNeuron] = tempWithBias;

  } else if (currentSubStep === 'sigmoid') {
    currentNeuron++;

    if (currentNeuron >= SIZES[currentLayer + 1]) {
      currentLayer++;
      currentNeuron = 0;
    }

    currentSubStep = 'idle';
  }

  drawNetwork();
  displayLayerValues();
  displayCalculation();
}

function stepBackprop() {
  if (currentSubStep === 'start_error') {
    // Initialize for first neuron error calculation
    currentNeuron = 0;
    currentSubStep = 'calc_error_neuron';

    if (!errors[currentLayer - 1]) {
      errors[currentLayer - 1] = [];
      biasGradients[currentLayer - 1] = [];
    }

  } else if (currentSubStep === 'calc_error_neuron') {
    // Calculate error for one output neuron
    const output = layerValues[currentLayer][currentNeuron];
    const target = currentTarget[currentNeuron];
    const z = weightedSums[currentLayer - 1][currentNeuron];
    const sigmoidDeriv = sigmoidDerivative(z);

    // Calculate cost and its derivative
    const costValue = calculateLoss(output, target);
    const costDerivative = output - target; // ∂L/∂a = ∂/∂a[0.5*(a - y)²] = (a - y)

    // Error signal (gradient) = ∂L/∂z = (∂L/∂a) · (∂a/∂z) = costDerivative * sigmoid'(z)
    const error = costDerivative * sigmoidDeriv;

    errors[currentLayer - 1][currentNeuron] = error;
    biasGradients[currentLayer - 1][currentNeuron] = error;

    // Store for display
    tempCostValue = costValue;
    tempCostDerivative = costDerivative;
    tempOutputError = output - target;
    tempSigmoidDeriv = sigmoidDeriv;
    tempError = error;

    currentNeuron++;

    if (currentNeuron >= SIZES[currentLayer]) {
      currentNeuron = 0;
      currentWeightIdx = 0;
      currentSubStep = 'calc_weight_grad_setup';
    }

  } else if (currentSubStep === 'calc_weight_grad_setup') {
    // Setup for weight gradient calculation
    const layerIdx = currentLayer - 1;
    if (!weightGradients[layerIdx]) {
      weightGradients[layerIdx] = [];
    }
    currentSubStep = 'calc_weight_grad_neuron';

  } else if (currentSubStep === 'calc_weight_grad_neuron') {
    // Calculate weight gradients for one neuron in current layer
    const layerIdx = currentLayer - 1;
    const error = errors[layerIdx][currentNeuron];

    if (!weightGradients[layerIdx][currentNeuron]) {
      weightGradients[layerIdx][currentNeuron] = [];
    }

    // Calculate gradient for each weight connected to this neuron
    for (let i = 0; i < SIZES[currentLayer - 1]; i++) {
      const activation = layerValues[currentLayer - 1][i];
      weightGradients[layerIdx][currentNeuron][i] = error * activation;
    }

    currentNeuron++;

    if (currentNeuron >= SIZES[currentLayer]) {
      // Done with this layer's weight gradients
      currentLayer--;

      if (currentLayer === 0) {
        phase = 'update';
        updateStatus('✅ Backpropagation complete! Ready to update weights.');
      } else {
        currentNeuron = 0;
        currentSubStep = 'propagate_error_neuron';
      }
    }

  } else if (currentSubStep === 'propagate_error_neuron') {
    // Propagate error to one neuron in previous layer
    const nextLayerIdx = currentLayer;

    if (!errors[currentLayer - 1]) {
      errors[currentLayer - 1] = [];
      biasGradients[currentLayer - 1] = [];
    }

    let errorSum = 0;
    tempErrorContributions = [];

    // Sum up weighted errors from next layer
    for (let j = 0; j < SIZES[currentLayer + 1]; j++) {
      const weight = weights[currentLayer][j][currentNeuron];
      const nextError = errors[nextLayerIdx][j];
      const contribution = weight * nextError;
      errorSum += contribution;
      tempErrorContributions.push({ weight, nextError, contribution });
    }

    const z = weightedSums[currentLayer - 1][currentNeuron];
    const sigmoidDeriv = sigmoidDerivative(z);
    const finalError = errorSum * sigmoidDeriv;

    errors[currentLayer - 1][currentNeuron] = finalError;
    biasGradients[currentLayer - 1][currentNeuron] = finalError;

    tempErrorSum = errorSum;
    tempSigmoidDeriv = sigmoidDeriv;
    tempError = finalError;

    currentNeuron++;

    if (currentNeuron >= SIZES[currentLayer]) {
      currentNeuron = 0;
      currentSubStep = 'calc_weight_grad_setup';
    }
  }

  drawNetwork();
  displayLayerValues();
  displayCalculation();
}

function stepUpdate() {
  // Take snapshots of weights and biases before updating so we can visualize the change
  oldWeights = weights.map(layer =>
    layer.map(neuronWeights => neuronWeights.slice())
  );
  oldBiases = biases.map(layerBiases => layerBiases.slice());

  // Update all weights and biases
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      for (let k = 0; k < weights[i][j].length; k++) {
        weights[i][j][k] -= LEARNING_RATE * weightGradients[i][j][k];
      }
      biases[i][j] -= LEARNING_RATE * biasGradients[i][j];
    }
  }

  phase = 'complete';
  updateStatus('🎉 Weights and biases updated! Network has learned from this example.');

  drawNetwork();
  displayLayerValues();
  displayCalculation();
}

window.skipToBackprop = function () {
  if (phase !== 'feedforward') {
    updateStatus('⚠️ Already past feedforward phase!');
    return;
  }

  // Complete feedforward quickly
  while (phase === 'feedforward') {
    stepFeedforward();
  }

  updateStatus('⚡ Skipped to backpropagation!');
}

window.autoPlay = function () {
  if (autoPlayInterval) return;

  autoPlayInterval = setInterval(() => {
    if (phase === 'complete') {
      stopAuto();
      return;
    }
    nextStep();
  }, 600);

  updateStatus('⏩ Auto-playing...');
}

window.stopAuto = function () {
  if (autoPlayInterval) {
    clearInterval(autoPlayInterval);
    autoPlayInterval = null;
  }
}

function updateStatus(message) {
  // Status bar removed - function kept for compatibility
}

function drawNetwork() {
  const canvas = document.getElementById('networkCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const layerSpacing = canvas.width / (SIZES.length + 1);

  // Draw connections
  for (let l = 0; l < SIZES.length - 1; l++) {
    const x1 = layerSpacing * (l + 1);
    const x2 = layerSpacing * (l + 2);
    const spacing1 = canvas.height / (SIZES[l] + 1);
    const spacing2 = canvas.height / (SIZES[l + 1] + 1);

    for (let i = 0; i < SIZES[l]; i++) {
      for (let j = 0; j < SIZES[l + 1]; j++) {
        const y1 = spacing1 * (i + 1);
        const y2 = spacing2 * (j + 1);

        const weight = weights[l][j][i];
        const intensity = Math.min(Math.abs(weight), 1);

        let isActive = false;
        let isBackprop = false;

        // Feedforward highlighting: current neuron collecting its weighted inputs
        if (
          phase === 'feedforward' &&
          l === currentLayer &&
          j === currentNeuron &&
          (currentSubStep === 'weighted_sum' || currentSubStep === 'add_bias')
        ) {
          isActive = true;
        }

        // Backprop highlighting:
        // - During weight-gradient calc, show all incoming connections to the neuron whose ∂L/∂w we are computing.
        // - During error propagation to previous layer, show all outgoing connections from the neuron whose error we are computing.
        if (phase === 'backprop') {
          // We are computing gradients for weights in layer (currentLayer - 1)
          if (
            currentSubStep === 'calc_weight_grad_neuron' &&
            l === currentLayer - 1 &&
            j === currentNeuron
          ) {
            isBackprop = true;
          }

          // We are propagating error from next layer back to the current layer
          if (
            currentSubStep === 'propagate_error_neuron' &&
            l === currentLayer &&
            i === currentNeuron
          ) {
            isBackprop = true;
          }
        }

        const color = isActive ? 'rgba(237, 137, 54, 0.9)' :
          isBackprop ? 'rgba(245, 101, 101, 0.8)' :
            weight > 0 ? `rgba(72, 187, 120, ${intensity * 0.6})` :
              `rgba(245, 101, 101, ${intensity * 0.6})`;

        ctx.strokeStyle = color;
        ctx.lineWidth = isActive ? 6 : isBackprop ? 4 : 2 + intensity * 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }

  // Draw neurons
  for (let l = 0; l < SIZES.length; l++) {
    const x = layerSpacing * (l + 1);
    const spacing = canvas.height / (SIZES[l] + 1);

    for (let i = 0; i < SIZES[l]; i++) {
      const y = spacing * (i + 1);

      const value = layerValues[l][i];
      const isEmpty = value === null;

      let isActive = false;
      let hasError = false;

      if (phase === 'feedforward' && l === currentLayer + 1 && i === currentNeuron &&
        currentSubStep !== 'idle') {
        isActive = true;
      }

      if (phase === 'backprop' && errors[l - 1] && errors[l - 1][i] !== undefined) {
        hasError = true;
      }

      if (isEmpty) {
        ctx.fillStyle = '#e2e8f0';
        ctx.strokeStyle = '#a0aec0';
        ctx.setLineDash([5, 5]);
      } else if (isActive) {
        ctx.fillStyle = '#fef5e7';
        ctx.strokeStyle = '#ed8936';
        ctx.setLineDash([]);
      } else if (hasError) {
        ctx.fillStyle = '#fed7d7';
        ctx.strokeStyle = '#f56565';
        ctx.setLineDash([]);
      } else {
        const intensity = value;
        ctx.fillStyle = `rgb(${255 * (1 - intensity)}, ${255 * intensity}, ${100 + 155 * intensity})`;
        ctx.strokeStyle = '#2d3748';
        ctx.setLineDash([]);
      }

      ctx.lineWidth = isActive ? 5 : hasError ? 4 : 3;

      ctx.beginPath();
      ctx.arc(x, y, 32, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw value
      ctx.fillStyle = isEmpty ? '#a0aec0' : '#000';
      ctx.font = isEmpty ? '14px sans-serif' : 'bold 13px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(isEmpty ? '?' : value.toFixed(3), x, y);

      // Draw error if in backprop
      if (hasError && errors[l - 1][i]) {
        ctx.fillStyle = '#e53e3e';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText(`ε=${errors[l - 1][i].toFixed(2)}`, x, y + 45);
      }
    }

    // Layer labels
    ctx.fillStyle = '#2d3748';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Layer ${l + 1}`, x, 35);
    ctx.font = '14px sans-serif';
    ctx.fillStyle = '#718096';
    ctx.fillText(LAYER_NAMES[l], x, 55);
  }
}

function renderWeightMatrix() {
  // Determine which weight layer we are focusing on:
  // - During feedforward we use the currentLayer (from layer → layer+1)
  // - During backprop we focus on the layer whose gradients / errors we are computing
  const layerIdx = phase === 'backprop' ? currentLayer - 1 : currentLayer;

  if (phase === 'complete' || layerIdx < 0 || layerIdx >= weights.length) return '';

  const matrix = weights[layerIdx];
  const biasVec = biases[layerIdx];

  let html = '<div class="matrices-display">';

  // Weight matrix
  html += '<div class="matrix-section">';
  html += `<div class="matrix-label">⚖️ Weight Matrix [Layer ${layerIdx + 1} → ${layerIdx + 2}]</div>`;
  html += '<div class="matrix-grid">';

  for (let row = 0; row < matrix.length; row++) {
    html += '<div class="matrix-row">';
    for (let col = 0; col < matrix[row].length; col++) {
      // Highlight logic:
      // - Feedforward: highlight the row for the neuron we are currently activating.
      // - Backprop (weight gradients): highlight the row for the neuron whose dC/dw we are computing.
      const isFeedforwardHighlight =
        phase === 'feedforward' &&
        layerIdx === currentLayer &&
        row === currentNeuron &&
        (currentSubStep === 'weighted_sum' || currentSubStep === 'add_bias');

      const isBackpropWeightHighlight =
        phase === 'backprop' &&
        layerIdx === currentLayer - 1 &&
        currentSubStep === 'calc_weight_grad_neuron' &&
        row === currentNeuron;

      const isHighlight = isFeedforwardHighlight || isBackpropWeightHighlight;

      const val = matrix[row][col];
      const intensity = Math.min(Math.abs(val), 1);
      const bgColor = val > 0
        ? `rgba(72, 187, 120, ${intensity * 0.35})`
        : `rgba(245, 101, 101, ${intensity * 0.35})`;

      html += `<div class="matrix-cell ${isHighlight ? 'highlight' : ''}" style="background: ${bgColor};">
        ${val.toFixed(3)}
      </div>`;
    }
    html += '</div>';
  }

  html += '</div></div>';

  // Bias vector
  html += '<div class="matrix-section">';
  html += `<div class="matrix-label">➕ Bias Vector [Layer ${layerIdx + 2}]</div>`;
  html += '<div class="bias-vector">';

  for (let i = 0; i < biasVec.length; i++) {
    const isHighlight = (phase === 'feedforward' && i === currentNeuron &&
      (currentSubStep === 'add_bias' || currentSubStep === 'sigmoid'));
    const val = biasVec[i];
    const intensity = Math.min(Math.abs(val), 1);
    const bgColor = val > 0
      ? `rgba(72, 187, 120, ${intensity * 0.35})`
      : `rgba(245, 101, 101, ${intensity * 0.35})`;

    html += `<div class="bias-cell ${isHighlight ? 'highlight' : ''}" style="background: ${bgColor};">
      b[${i}] = ${val.toFixed(3)}
    </div>`;
  }

  html += '</div></div>';
  html += '</div>';

  return html;
}

// Show side-by-side comparison of weights and biases before vs after the update
function renderUpdateComparison() {
  if (!oldWeights || !oldBiases) return '';

  let html = '<div>';

  // First show all weight matrices
  html += '<div style="margin-bottom: 40px;">';
  html += '<h3 style="color: #2d3748; margin-bottom: 20px; text-align: center;">⚖️ Weight Matrices</h3>';

  for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
    const oldMatrix = oldWeights[layerIdx];
    const newMatrix = weights[layerIdx];

    html += `<div class="layer-section" style="margin-bottom: 30px;">`;
    html += `<h4 style="text-align: center; margin-bottom: 15px;">Layer ${layerIdx + 1} → ${layerIdx + 2}</h4>`;
    html += `<div style="display: flex; gap: 20px; align-items: flex-start; justify-content: center;">`;

    // Old weights table
    html += '<div style="flex: 1; min-width: 0;">';
    html += '<div class="matrix-section">';
    html += `<div class="matrix-label">Before update (w<sub>old</sub>)</div>`;
    html += '<div class="matrix-grid">';
    for (let row = 0; row < oldMatrix.length; row++) {
      html += '<div class="matrix-row">';
      for (let col = 0; col < oldMatrix[row].length; col++) {
        const oldVal = oldMatrix[row][col];
        html += `<div class="matrix-cell" style="background: #f7fafc;">
          ${oldVal.toFixed(3)}
        </div>`;
      }
      html += '</div>';
    }
    html += '</div></div></div>';

    // New weights table with color coding
    html += '<div style="flex: 1; min-width: 0;">';
    html += '<div class="matrix-section">';
    html += `<div class="matrix-label">After update (w<sub>new</sub>)</div>`;
    html += '<div class="matrix-grid">';
    for (let row = 0; row < newMatrix.length; row++) {
      html += '<div class="matrix-row">';
      for (let col = 0; col < newMatrix[row].length; col++) {
        const oldVal = oldMatrix[row][col];
        const newVal = newMatrix[row][col];
        const delta = newVal - oldVal;

        let bgColor = '#ffffff'; // white for no change
        if (Math.abs(delta) > 0.0001) {
          if (delta > 0) {
            bgColor = '#c6f6d5'; // green for increase
          } else {
            bgColor = '#fed7d7'; // red for decrease
          }
        }

        html += `<div class="matrix-cell" style="background: ${bgColor};">
          ${newVal.toFixed(3)}
        </div>`;
      }
      html += '</div>';
    }
    html += '</div></div></div>';

    html += '</div></div>'; // end flex and layer-section
  }
  html += '</div>'; // end weights section

  // Then show all bias vectors
  html += '<div style="margin-bottom: 20px;">';
  html += '<h3 style="color: #2d3748; margin-bottom: 20px; text-align: center;">➕ Bias Vectors</h3>';

  for (let layerIdx = 0; layerIdx < biases.length; layerIdx++) {
    const oldBiasVec = oldBiases[layerIdx];
    const newBiasVec = biases[layerIdx];

    html += `<div class="layer-section" style="margin-bottom: 30px;">`;
    html += `<h4 style="text-align: center; margin-bottom: 15px;">Layer ${layerIdx + 2}</h4>`;
    html += `<div style="display: flex; gap: 20px; align-items: flex-start; justify-content: center;">`;

    // Old biases
    html += '<div style="flex: 1; min-width: 0;">';
    html += '<div class="matrix-section">';
    html += `<div class="matrix-label">Before update (b<sub>old</sub>)</div>`;
    html += '<div class="bias-vector">';
    for (let i = 0; i < oldBiasVec.length; i++) {
      const oldB = oldBiasVec[i];
      html += `<div class="bias-cell" style="background: #f7fafc;">
        b<sub>${i}</sub> = ${oldB.toFixed(3)}
      </div>`;
    }
    html += '</div></div></div>';

    // New biases with color coding
    html += '<div style="flex: 1; min-width: 0;">';
    html += '<div class="matrix-section">';
    html += `<div class="matrix-label">After update (b<sub>new</sub>)</div>`;
    html += '<div class="bias-vector">';
    for (let i = 0; i < newBiasVec.length; i++) {
      const oldB = oldBiasVec[i];
      const newB = newBiasVec[i];
      const deltaB = newB - oldB;

      let bgColorB = '#ffffff'; // white for no change
      if (Math.abs(deltaB) > 0.0001) {
        if (deltaB > 0) {
          bgColorB = '#c6f6d5'; // green for increase
        } else {
          bgColorB = '#fed7d7'; // red for decrease
        }
      }

      html += `<div class="bias-cell" style="background: ${bgColorB};">
        b<sub>${i}</sub> = ${newB.toFixed(3)}
      </div>`;
    }
    html += '</div></div></div>';

    html += '</div></div>'; // end flex and layer-section
  }
  html += '</div>'; // end biases section

  html += '</div>'; // end main container

  return html;
}

function displayCalculation() {
  const container = document.getElementById('calculationContent');
  const networkPanel = document.getElementById('networkPanel');
  const calculationPanel = document.getElementById('calculationPanel');
  const mainGrid = document.getElementById('mainGrid');

  if (phase === 'feedforward') {
    // Show both panels in grid
    if (networkPanel) networkPanel.style.display = 'block';
    if (mainGrid) mainGrid.style.display = 'grid';
    if (calculationPanel) {
      calculationPanel.style.maxWidth = '';
      calculationPanel.style.gridColumn = '';
      const panelHeader = calculationPanel.querySelector('h3');
      if (panelHeader) panelHeader.style.display = 'block';
    }
    displayFeedforwardCalc(container);
  } else if (phase === 'backprop') {
    // Show both panels in grid
    if (networkPanel) networkPanel.style.display = 'block';
    if (mainGrid) mainGrid.style.display = 'grid';
    if (calculationPanel) {
      calculationPanel.style.maxWidth = '';
      calculationPanel.style.gridColumn = '';
      const panelHeader = calculationPanel.querySelector('h3');
      if (panelHeader) panelHeader.style.display = 'block';
    }
    displayBackpropCalc(container);
  } else if (phase === 'update') {
    // Show both panels in grid
    if (networkPanel) networkPanel.style.display = 'block';
    if (mainGrid) mainGrid.style.display = 'grid';
    if (calculationPanel) {
      calculationPanel.style.maxWidth = '';
      calculationPanel.style.gridColumn = '';
      const panelHeader = calculationPanel.querySelector('h3');
      if (panelHeader) panelHeader.style.display = 'block';
    }
    displayUpdateCalc(container);
  } else if (phase === 'complete') {
    // Hide network panel and make calculation panel full width, but keep grid layout
    if (networkPanel) networkPanel.style.display = 'none';
    if (mainGrid) mainGrid.style.display = 'grid';
    if (calculationPanel) {
      calculationPanel.style.maxWidth = '100%';
      calculationPanel.style.gridColumn = '1 / -1';
      // Hide the panel header
      const panelHeader = calculationPanel.querySelector('h3');
      if (panelHeader) panelHeader.style.display = 'none';
    }

    container.innerHTML = renderUpdateComparison();
  }
}

function displayFeedforwardCalc(container) {
  if (currentSubStep === 'idle' && currentLayer < SIZES.length - 1) {
    container.innerHTML = `
      <div style="min-height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div class="neuron-info">
          <strong>Ready to calculate:</strong><br>
          Layer ${currentLayer + 2} (${LAYER_NAMES[currentLayer + 1]}), Neuron ${currentNeuron}
        </div>
        <p style="color: #718096; margin-top: 15px; font-size: 15px;">Click "Next Step" to start weighted sum calculation.</p>
        ${renderWeightMatrix()}
      </div>
    `;
  } else if (currentSubStep === 'weighted_sum') {
    let html = `
      <div class="calculation-box">
        <strong>📊 STEP 1: Weighted Sum</strong>
        <div class="formula">
          z = Σ(weight<sub>i</sub> × input<sub>i</sub>)
        </div>
        <p style="margin: 15px 0;">Calculating for <span class="step-highlight">Layer ${currentLayer + 2}, Neuron ${currentNeuron}</span>:</p>
    `;

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      const weight = weights[currentLayer][currentNeuron][i];
      const input = layerValues[currentLayer][i];
      const product = weight * input;
      html += `<div style="margin: 10px 0; padding-left: 15px; font-size: 15px;">
        (${weight.toFixed(3)} × ${input.toFixed(3)}) = <strong>${product.toFixed(3)}</strong>
      </div>`;
    }

    html += `
        <div style="margin-top: 18px; padding: 12px; background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); border-radius: 8px;">
          <strong style="color: #234e52;">Sum = ${tempWeightedSum.toFixed(4)}</strong>
        </div>
      </div>
      ${renderWeightMatrix()}
    `;
    container.innerHTML = html;

  } else if (currentSubStep === 'add_bias') {
    const bias = biases[currentLayer][currentNeuron];
    container.innerHTML = `
      <div class="calculation-box">
        <strong>➕ STEP 2: Add Bias</strong>
        <div class="formula">
          z + bias
        </div>
        <p style="margin: 15px 0;">Adding bias to weighted sum:</p>
        <div style="padding-left: 15px; font-size: 17px;">
          ${tempWeightedSum.toFixed(4)} + ${bias.toFixed(4)} = <strong>${tempWithBias.toFixed(4)}</strong>
        </div>
      </div>
      ${renderWeightMatrix()}
    `;

  } else if (currentSubStep === 'sigmoid') {
    const result = layerValues[currentLayer + 1][currentNeuron];
    container.innerHTML = `
      <div class="calculation-box">
        <strong>🔄 STEP 3: Apply Sigmoid</strong>
        <div class="formula">
          σ(z) = 1 / (1 + e<sup>-z</sup>)
        </div>
        <p style="margin: 15px 0;">Squashing value to range [0, 1]:</p>
        <div style="padding-left: 15px; font-size: 17px;">
          σ(${tempWithBias.toFixed(4)}) = <strong style="color: #48bb78;">${result.toFixed(4)}</strong>
        </div>
        <div style="margin-top: 18px; padding: 12px; background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); border-radius: 8px; color: #22543d;">
          ✅ Neuron ${currentNeuron} in Layer ${currentLayer + 2} complete!
        </div>
      </div>
      ${renderWeightMatrix()}
    `;
  }
}

function displayBackpropCalc(container) {
  if (currentSubStep === 'start_error') {
    container.innerHTML = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">🔙 Starting Backpropagation</strong>
        <p style="margin: 15px 0; font-size: 16px;">
          We'll now work backwards through the network to calculate how much each weight and bias contributed to the error.
        </p>
        <div style="padding: 12px; background: linear-gradient(135deg, #fef5e7 0%, #fdebd0 100%); border-radius: 8px; color: #7d6608; margin-top: 15px;">
          📚 <strong>What is backpropagation?</strong><br>
          It's the process of calculating gradients (how much to adjust each weight/bias) by propagating the error backward through the network.
        </div>
        <p style="margin-top: 15px; color: #718096;">Click "Next Step" to calculate error for the first output neuron.</p>
      </div>
    `;

  } else if (currentSubStep === 'calc_error_neuron') {
    const neuronIdx = currentNeuron - 1; // Already incremented
    const output = layerValues[currentLayer][neuronIdx];
    const target = currentTarget[neuronIdx];
    const z = weightedSums[currentLayer - 1][neuronIdx];

    let html = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ STEP 1: Calculate Gradient (Error) for Output Neuron ${neuronIdx}</strong>
        
        <div style="margin: 20px 0; padding: 15px; background: #fef5e7; border-radius: 8px;">
          <strong style="color: #92400e;">📖 What we're doing (notation):</strong><br>
          We use the <strong>Chain Rule</strong> with the standard symbols:<br>
          <code>L</code> = loss, <code>a</code> = activation/output of this neuron, <code>z</code> = weighted sum before activation, <code>w</code> = weight.<br>
          We want <strong>∂L/∂z</strong> for this neuron.
        </div>
        
        <div style="margin: 20px 0; padding: 12px; background: #e0e7ff; border-radius: 8px;">
          <strong>📐 Chain Rule:</strong><br>
          <code style="font-size: 14px;">∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)</code><br>
          Here we first compute <code>∂L/∂a</code> and <code>∂a/∂z</code>, then later use them to get <code>∂L/∂w</code>.
        </div>
        
        <div style="margin: 20px 0;">
          <div style="padding: 12px; background: #fef3c7; border-radius: 8px; margin: 10px 0;">
            <strong>📊 Loss function (MSE):</strong><br>
            <span style="font-size: 15px; margin-left: 20px;">
              L = 0.5 × (a - y)²
            </span><br>
            <span style="font-size: 15px; margin-left: 20px;">
              L = 0.5 × (${output.toFixed(4)} - ${target.toFixed(4)})² = <strong style="color: #dc2626;">${tempCostValue.toFixed(6)}</strong>
            </span>
          </div>
          
          <div style="padding: 12px; background: #f0f9ff; border-radius: 8px; margin: 10px 0;">
            <strong>Part 1: ∂L/∂a (loss w.r.t. activation)</strong><br>
            <span style="font-size: 15px; margin-left: 20px;">
              ∂/∂a [0.5 × (a - y)²] = (a - y)
            </span><br>
            <span style="font-size: 16px; margin-left: 20px;">
              = ${output.toFixed(4)} - ${target.toFixed(4)} = <strong style="color: #dc2626;">${tempCostDerivative.toFixed(4)}</strong>
            </span><br>
            <span style="font-size: 13px; color: #64748b; margin-left: 20px;">
              (How much the loss changes if we nudge this neuron's activation)
            </span>
          </div>
          
          <div style="padding: 12px; background: #f0f9ff; border-radius: 8px; margin: 10px 0;">
            <strong>Part 2: ∂a/∂z = σ'(z) (activation w.r.t. weighted sum)</strong><br>
            <span style="font-size: 16px; margin-left: 20px;">
              σ'(${z.toFixed(4)}) = <strong style="color: #2563eb;">${tempSigmoidDeriv.toFixed(4)}</strong>
            </span><br>
            <span style="font-size: 13px; color: #64748b; margin-left: 20px;">
              (How much the activation changes if we nudge z)
            </span>
          </div>
          
          <div style="padding: 14px; background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); border-radius: 8px; margin: 10px 0;">
            <strong>⛓️ Chain Rule Result: ∂L/∂z (this neuron's gradient)</strong><br>
            <span style="font-size: 16px; margin-left: 20px;">
              ∂L/∂z = (∂L/∂a) · (∂a/∂z) = ${tempCostDerivative.toFixed(4)} × ${tempSigmoidDeriv.toFixed(4)} = <strong style="color: #991b1b; font-size: 18px;">${tempError.toFixed(4)}</strong>
            </span><br>
            <span style="font-size: 13px; color: #7f1d1d; margin-left: 20px;">
              ✅ This is often written as δ = ∂L/∂z and will be used to get ∂L/∂w.
            </span>
          </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #fef3c7; border-radius: 8px; color: #78350f;">
          💡 <strong>Simple explanation:</strong> The gradient δ = ${tempError.toFixed(4)} is exactly ∂L/∂z. Multiply it by inputs later and you get ∂L/∂w.
        </div>
      </div>
    `;
    container.innerHTML = html;

  } else if (currentSubStep === 'calc_weight_grad_setup') {
    container.innerHTML = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ Ready to Calculate Weight Gradients (∂L/∂w)</strong>
        <p style="margin: 15px 0; font-size: 16px;">
          Now that we know <strong>∂L/∂z</strong> (δ) for each neuron, we can calculate <strong>∂L/∂w</strong> (how the loss changes with each weight).
        </p>
        <div style="padding: 12px; background: #e0e7ff; border-radius: 8px; margin: 15px 0;">
          <strong>📐 Chain Rule for Weights (final step):</strong><br>
          We already have δ = ∂L/∂z from the previous step.<br>
          <code style="font-size: 14px;">∂L/∂w = (∂L/∂z) · (∂z/∂w)</code><br>
          <span style="font-size: 13px; color: #3730a3;">
            Since z = Σ(w · a<sub>prev</sub>) + b, we have: ∂z/∂w = a<sub>prev</sub>.
          </span><br>
          <code style="font-size: 14px;">Therefore: ∂L/∂w = δ · a<sub>prev</sub> = gradient × activation</code>
        </div>
        <div style="padding: 12px; background: linear-gradient(135deg, #fef5e7 0%, #fdebd0 100%); border-radius: 8px; color: #7d6608;">
          📚 <strong>Simple Formula:</strong><br>
          ∂L/∂w = (neuron's gradient δ) × (activation from previous layer a<sub>prev</sub>)<br><br>
          This tells us how much to adjust each weight to reduce the loss.
        </div>
        <p style="margin-top: 15px; color: #718096;">Click "Next Step" to calculate gradients for the first neuron.</p>
      </div>
    `;

  } else if (currentSubStep === 'calc_weight_grad_neuron') {
    const neuronIdx = currentNeuron - 1; // Already incremented
    const layerIdx = currentLayer - 1;
    const neuronGradient = errors[layerIdx][neuronIdx];

    let html = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ STEP 2: Calculate ∂L/∂w for All Weights to Neuron ${neuronIdx}</strong>
        
        <div style="margin: 20px 0; padding: 15px; background: #fef5e7; border-radius: 8px;">
          <strong style="color: #92400e;">📖 What we're doing:</strong><br>
          For each weight connecting to this neuron, we calculate <strong>∂L/∂w</strong> (how the loss changes with that weight).
        </div>
        
        <div class="formula" style="margin: 20px 0;">
          ∂L/∂w = (∂L/∂z) · (∂z/∂w) = δ × a<sub>prev</sub>
        </div>
        
        <div style="margin: 20px 0;">
          <div style="padding: 12px; background: #ede9fe; border-radius: 8px; margin-bottom: 15px;">
            <strong>This neuron's gradient δ = ∂L/∂z:</strong> <span style="color: #dc2626; font-size: 16px;">${neuronGradient.toFixed(4)}</span>
          </div>
          
          <strong>Calculating ∂L/∂w for each weight:</strong>
    `;

    for (let i = 0; i < SIZES[currentLayer - 1]; i++) {
      const activation = layerValues[currentLayer - 1][i];
      const weightGradient = weightGradients[layerIdx][neuronIdx][i];
      html += `
        <div style="padding: 10px; background: #f0f9ff; border-radius: 8px; margin: 8px 0;">
          <strong>Weight from Neuron ${i} in previous layer:</strong><br>
            <span style="margin-left: 20px; font-size: 15px;">
            ∂L/∂w = δ × a<sub>prev</sub> = ${neuronGradient.toFixed(4)} × ${activation.toFixed(4)} = <strong style="color: #1e40af;">${weightGradient.toFixed(4)}</strong>
          </span><br>
          <span style="font-size: 13px; color: #64748b; margin-left: 20px;">
            (gradient × activation from previous layer)
          </span>
        </div>
      `;
    }

    html += `
        </div>
        
        <div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); border-radius: 8px; color: #7f1d1d;">
          ✅ All weight gradients (dC/dw) calculated for Neuron ${neuronIdx}!
        </div>
        
        <div style="margin-top: 12px; padding: 10px; background: #fef3c7; border-radius: 8px; color: #78350f;">
          💡 <strong>Remember:</strong> These gradients are ∂L/∂w. In the update step, we'll use: <strong>w_new = w_old - lr × ∂L/∂w</strong>
        </div>
      </div>
    `;
    container.innerHTML = html;

  } else if (currentSubStep === 'propagate_error_neuron') {
    const neuronIdx = currentNeuron - 1; // Already incremented

    let html = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ STEP 3: Propagate Error to Layer ${currentLayer + 1}, Neuron ${neuronIdx}</strong>
        
        <div style="margin: 20px 0; padding: 15px; background: #fef5e7; border-radius: 8px;">
          <strong style="color: #92400e;">📖 What we're doing:</strong><br>
          Calculate how much error this neuron contributed to the next layer's errors. We look at all weights going forward and their errors.
        </div>
        
        <div class="formula" style="margin: 20px 0;">
          error = Σ(weight × next_error) × σ'(z)
        </div>
        
        <div style="margin: 20px 0;">
          <strong>Step 3a: Sum up weighted errors from next layer</strong>
    `;

    for (let j = 0; j < tempErrorContributions.length; j++) {
      const contrib = tempErrorContributions[j];
      html += `
        <div style="padding: 10px; background: #f0f9ff; border-radius: 8px; margin: 8px 0;">
          <strong>From next layer Neuron ${j}:</strong><br>
          <span style="margin-left: 20px; font-size: 15px;">
            weight: ${contrib.weight.toFixed(4)} × error: ${contrib.nextError.toFixed(4)} = <strong>${contrib.contribution.toFixed(4)}</strong>
          </span>
        </div>
      `;
    }

    html += `
          <div style="padding: 12px; background: #dbeafe; border-radius: 8px; margin: 10px 0;">
            <strong>Sum of contributions:</strong> <span style="color: #1e40af; font-size: 16px;">${tempErrorSum.toFixed(4)}</span>
          </div>
        </div>
        
        <div style="margin: 20px 0;">
          <strong>Step 3b: Multiply by sigmoid derivative</strong>
          <div style="padding: 12px; background: #f0f9ff; border-radius: 8px; margin: 10px 0;">
            ${tempErrorSum.toFixed(4)} × σ'(z)[${tempSigmoidDeriv.toFixed(4)}] = <strong style="color: #dc2626; font-size: 18px;">${tempError.toFixed(4)}</strong>
          </div>
        </div>
        
        <div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); border-radius: 8px; color: #7f1d1d;">
          ✅ Error calculated for Neuron ${neuronIdx} in Layer ${currentLayer + 1}!
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #fef3c7; border-radius: 8px; color: #78350f;">
          💡 <strong>Simple explanation:</strong> This neuron's error is the weighted sum of all errors it contributed to in the next layer.
        </div>
      </div>
    `;
    container.innerHTML = html;
  }
}

function displayUpdateCalc(container) {
  let html = `
    <div class="calculation-box">
      <strong style="color: #38a169;">💾 FINAL STEP: Gradient Descent - Update All Weights & Biases</strong>
      
      <div style="margin: 20px 0; padding: 15px; background: #ecfdf5; border-radius: 8px;">
        <strong style="color: #065f46;">📖 What we're doing:</strong><br>
        Now we use <strong>Gradient Descent</strong>: moving each weight in the direction that reduces the cost. We subtract the gradient (times learning rate) from each weight.
      </div>
      
      <div style="margin: 15px 0; padding: 12px; background: #e0e7ff; border-radius: 8px;">
        <strong>📐 Gradient Descent Update Rule:</strong><br>
        <code style="font-size: 15px;">w<sub>new</sub> = w<sub>old</sub> - learning_rate × ∂L/∂w</code><br>
        <code style="font-size: 15px;">b<sub>new</sub> = b<sub>old</sub> - learning_rate × ∂L/∂b</code><br>
        <span style="font-size: 13px; color: #3730a3; margin-top: 8px; display: block;">
          The minus sign means we go <strong>downhill</strong> (opposite direction of gradient) to minimize cost!
        </span>
      </div>
      
      <div style="padding: 12px; background: #dbeafe; border-radius: 8px; margin: 15px 0;">
        <strong>Learning Rate (lr):</strong> <span style="color: #1e40af; font-size: 16px;">${LEARNING_RATE}</span><br>
        <span style="font-size: 14px; color: #64748b;">
          (Controls step size - too large and we overshoot, too small and learning is slow)
        </span>
      </div>
      
      <div style="margin: 20px 0;">
        <strong>Example weight updates from Layer 1 → Layer 2:</strong>
  `;

  // Show a few example weight updates
  if (weightGradients[0] && weightGradients[0][0]) {
    for (let i = 0; i < Math.min(3, weightGradients[0][0].length); i++) {
      const oldWeight = weights[0][0][i] + (LEARNING_RATE * weightGradients[0][0][i]); // Reverse the update to show old value
      const gradient = weightGradients[0][0][i];
      const newWeight = weights[0][0][i];
      const change = LEARNING_RATE * gradient;

      html += `
        <div style="padding: 10px; background: #f0fdf4; border-radius: 8px; margin: 8px 0;">
          <strong>Weight[0][0][${i}]:</strong><br>
          <div style="margin-left: 20px; font-size: 14px; margin-top: 5px;">
            <div>w<sub>old</sub> = ${oldWeight.toFixed(4)}</div>
            <div>∂L/∂w = ${gradient.toFixed(4)}</div>
            <div style="margin-top: 5px; padding: 8px; background: #fff; border-radius: 4px;">
              w<sub>new</sub> = ${oldWeight.toFixed(4)} - (${LEARNING_RATE} × ${gradient.toFixed(4)})<br>
              w<sub>new</sub> = ${oldWeight.toFixed(4)} - ${change.toFixed(4)}<br>
              w<sub>new</sub> = <strong style="color: #15803d; font-size: 16px;">${newWeight.toFixed(4)}</strong>
            </div>
          </div>
        </div>
      `;
    }
  }

  html += `
      </div>
      
      <div style="margin-top: 18px; padding: 12px; background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); border-radius: 8px; color: #22543d;">
        ✅ Successfully updated all ${weights.reduce((sum, layer) => sum + layer.reduce((s, neuron) => s + neuron.length, 0), 0)} weights and ${biases.reduce((sum, layer) => sum + layer.length, 0)} biases using gradient descent!
      </div>
      
      <div style="margin-top: 15px; padding: 10px; background: #fef3c7; border-radius: 8px; color: #78350f;">
        💡 <strong>Simple explanation:</strong> By subtracting (lr × gradient), we moved each weight down the "cost slope" to reduce the error. This is gradient descent!
      </div>
      
      <div style="margin-top: 15px; padding: 12px; background: #ede9fe; border-radius: 8px; color: #5b21b6;">
        🎓 <strong>What happens next?</strong><br>
        In real training, we repeat: Feedforward → Backprop → Update thousands of times with different examples. The network gradually minimizes the cost function and learns patterns!
      </div>
    </div>
  `;

  // Append full before/after matrices & biases comparison
  html += renderUpdateComparison();

  container.innerHTML = html;
}

// Layer values display removed - section no longer in UI
function displayLayerValues() {
  // Function kept for compatibility
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  resetNetwork();
});

