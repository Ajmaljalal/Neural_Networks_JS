import { sigmoid, sigmoidDerivative, randomGaussian } from './utils.js';

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

let currentInput = [0.5, 0.8, 0.3, 0.9];
let currentTarget = [1.0, 0.0];

let phase = 'idle'; // 'idle', 'feedforward', 'backprop', 'update'
let currentLayer = 0;
let currentNeuron = 0;
let currentSubStep = 'idle';
let tempWeightedSum = 0;
let tempWithBias = 0;
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
  updateProgress();
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
    currentSubStep = 'calc_error';
    updateStatus('✅ Feedforward complete! Starting backpropagation...');
    drawNetwork();
    displayLayerValues();
    displayCalculation();
    updateProgress();
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
  updateProgress();
}

function stepBackprop() {
  if (currentSubStep === 'calc_error') {
    // Calculate output layer errors
    errors[currentLayer - 1] = [];
    biasGradients[currentLayer - 1] = [];

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      const error = (layerValues[currentLayer][i] - currentTarget[i]) *
        sigmoidDerivative(weightedSums[currentLayer - 1][i]);
      errors[currentLayer - 1].push(error);
      biasGradients[currentLayer - 1].push(error);
    }

    currentSubStep = 'calc_weight_grad';

  } else if (currentSubStep === 'calc_weight_grad') {
    // Calculate weight gradients for current layer
    const layerIdx = currentLayer - 1;
    weightGradients[layerIdx] = [];

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      weightGradients[layerIdx].push(
        layerValues[currentLayer - 1].map(a => errors[layerIdx][i] * a)
      );
    }

    currentLayer--;

    if (currentLayer === 0) {
      phase = 'update';
      updateStatus('✅ Backpropagation complete! Ready to update weights.');
    } else {
      currentSubStep = 'propagate_error';
    }

  } else if (currentSubStep === 'propagate_error') {
    // Propagate errors to previous layer
    const nextLayerIdx = currentLayer;
    errors[currentLayer - 1] = [];
    biasGradients[currentLayer - 1] = [];

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      let error = 0;
      for (let j = 0; j < SIZES[currentLayer + 1]; j++) {
        error += weights[currentLayer][j][i] * errors[nextLayerIdx][j];
      }
      error *= sigmoidDerivative(weightedSums[currentLayer - 1][i]);
      errors[currentLayer - 1].push(error);
      biasGradients[currentLayer - 1].push(error);
    }

    currentSubStep = 'calc_weight_grad';
  }

  drawNetwork();
  displayLayerValues();
  displayCalculation();
  updateProgress();
}

function stepUpdate() {
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
  updateProgress();
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

function updateProgress() {
  let totalSteps = 0;
  let completedSteps = 0;

  // Feedforward steps
  const ffSteps = SIZES.slice(1).reduce((a, b) => a + b, 0) * 3;
  totalSteps += ffSteps;

  if (phase === 'feedforward') {
    const neuronsCompleted = SIZES.slice(1, currentLayer + 1).reduce((a, b) => a + b, 0) + currentNeuron;
    const substepProgress = ['idle', 'weighted_sum', 'add_bias', 'sigmoid'].indexOf(currentSubStep);
    completedSteps = neuronsCompleted * 3 + substepProgress;
  } else {
    completedSteps = ffSteps;
  }

  // Backprop steps
  const bpSteps = SIZES.length - 1;
  totalSteps += bpSteps;

  if (phase === 'backprop') {
    completedSteps += (SIZES.length - 1 - currentLayer);
  } else if (phase === 'update' || phase === 'complete') {
    completedSteps += bpSteps;
  }

  // Update step
  totalSteps += 1;
  if (phase === 'complete') {
    completedSteps += 1;
  }

  const percentage = Math.floor((completedSteps / totalSteps) * 100);
  const progressBar = document.getElementById('progress');
  const progressText = document.getElementById('progressText');
  progressBar.style.width = percentage + '%';
  progressText.textContent = percentage + '%';
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

        if (phase === 'feedforward' && l === currentLayer && j === currentNeuron &&
          (currentSubStep === 'weighted_sum' || currentSubStep === 'add_bias')) {
          isActive = true;
        }

        if (phase === 'backprop' && l === currentLayer - 1 && errors[l]) {
          isBackprop = true;
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
  if (phase === 'complete' || currentLayer >= SIZES.length - 1) return '';

  const layerIdx = phase === 'backprop' ? currentLayer - 1 : currentLayer;
  if (layerIdx < 0 || layerIdx >= weights.length) return '';

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
      const isHighlight = (phase === 'feedforward' && row === currentNeuron &&
        (currentSubStep === 'weighted_sum' || currentSubStep === 'add_bias'));
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

function displayCalculation() {
  const container = document.getElementById('calculationContent');

  if (phase === 'feedforward') {
    displayFeedforwardCalc(container);
  } else if (phase === 'backprop') {
    displayBackpropCalc(container);
  } else if (phase === 'update') {
    displayUpdateCalc(container);
  } else if (phase === 'complete') {
    container.innerHTML = `
      <div style="text-align: center; padding: 50px; color: #48bb78;">
        <h2 style="font-size: 2em; margin-bottom: 15px;">🎉 Training Complete!</h2>
        <p style="color: #718096; font-size: 1.1em;">
          The network has completed one full training cycle:<br>
          <strong>Feedforward → Backpropagation → Weight Update</strong>
        </p>
        <div style="margin-top: 25px; padding: 20px; background: #f7fafc; border-radius: 12px; display: inline-block;">
          <p style="color: #2d3748; margin: 5px 0;"><strong>📥 Input:</strong> [${currentInput.join(', ')}]</p>
          <p style="color: #2d3748; margin: 5px 0;"><strong>🎯 Target:</strong> [${currentTarget.join(', ')}]</p>
          <p style="color: #2d3748; margin: 5px 0;"><strong>📤 Output:</strong> [${layerValues[SIZES.length - 1].map(v => v.toFixed(3)).join(', ')}]</p>
        </div>
      </div>
    `;
  }
}

function displayFeedforwardCalc(container) {
  if (currentSubStep === 'idle' && currentLayer < SIZES.length - 1) {
    container.innerHTML = `
      <div class="neuron-info">
        <strong>🎯 Ready to calculate:</strong><br>
        Layer ${currentLayer + 2} (${LAYER_NAMES[currentLayer + 1]}), Neuron ${currentNeuron}
      </div>
      <p style="color: #718096; margin-top: 15px; font-size: 15px;">Click "Next Step" to start weighted sum calculation.</p>
      ${renderWeightMatrix()}
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
  if (currentSubStep === 'calc_error') {
    let html = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ BACKPROP STEP 1: Calculate Output Error</strong>
        <div class="formula">
          δ = (output - target) × σ'(z)
        </div>
        <p style="margin: 15px 0;">Computing error for <span class="step-highlight">Output Layer</span>:</p>
    `;

    for (let i = 0; i < SIZES[currentLayer]; i++) {
      const output = layerValues[currentLayer][i];
      const target = currentTarget[i];
      const error = errors[currentLayer - 1] ? errors[currentLayer - 1][i] : 0;
      html += `<div style="margin: 10px 0; padding-left: 15px; font-size: 15px;">
        Neuron ${i}: (${output.toFixed(3)} - ${target.toFixed(3)}) × σ'(z) = <strong style="color: #e53e3e;">${error.toFixed(4)}</strong>
      </div>`;
    }

    html += `</div>`;
    container.innerHTML = html;

  } else if (currentSubStep === 'calc_weight_grad') {
    container.innerHTML = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ BACKPROP STEP 2: Calculate Weight Gradients</strong>
        <div class="formula">
          ∂W = δ × activation<sub>prev</sub>
        </div>
        <p style="margin: 15px 0;">Computing gradients for weights connecting to Layer ${currentLayer + 1}.</p>
        <div style="padding: 12px; background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); border-radius: 8px; color: #742a2a;">
          ✅ Gradients calculated for all weights in this layer!
        </div>
      </div>
    `;

  } else if (currentSubStep === 'propagate_error') {
    container.innerHTML = `
      <div class="calculation-box">
        <strong style="color: #e53e3e;">◀️ BACKPROP STEP 3: Propagate Error Backward</strong>
        <div class="formula">
          δ<sub>l</sub> = (W<sup>T</sup> × δ<sub>l+1</sub>) ⊙ σ'(z)
        </div>
        <p style="margin: 15px 0;">Propagating error from Layer ${currentLayer + 2} to Layer ${currentLayer + 1}.</p>
        <div style="padding: 12px; background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); border-radius: 8px; color: #742a2a;">
          Errors distributed based on weight contributions.
        </div>
      </div>
    `;
  }
}

function displayUpdateCalc(container) {
  container.innerHTML = `
    <div class="calculation-box">
      <strong style="color: #38a169;">💾 UPDATE STEP: Adjust Weights & Biases</strong>
      <div class="formula">
        W<sub>new</sub> = W<sub>old</sub> - learning_rate × ∂W<br>
        b<sub>new</sub> = b<sub>old</sub> - learning_rate × ∂b
      </div>
      <p style="margin: 15px 0;">Updating all weights and biases using:</p>
      <div style="padding-left: 15px; font-size: 16px;">
        <p><strong>Learning Rate:</strong> ${LEARNING_RATE}</p>
      </div>
      <div style="margin-top: 18px; padding: 12px; background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); border-radius: 8px; color: #22543d;">
        Ready to update! Click "Next Step" to apply changes.
      </div>
    </div>
  `;
}

// Layer values display removed - section no longer in UI
function displayLayerValues() {
  // Function kept for compatibility
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  resetNetwork();
});

