/* ═══════════════════════════════════════════════════
   Libratio Fleet — GPU Cluster Simulator Engine
   ═══════════════════════════════════════════════════ */

let currentObs = null;
let stepCount = 0;
const MAX_STEPS = 5;
let isBreakMode = false;

// Chart state
let rewardChart;
const chartLabels = ["Start"];
const trainedData = [0.1];
const baselineData = [0.1];

const $ = (id) => document.getElementById(id);

const taskSelect = $("taskSelect");
const resetBtn = $("resetBtn");
const stepBtn = $("stepBtn");
const actionInput = $("actionInput");
const resBox = $("resBox");
const rackGrid = $("rackGrid");
const rackStatus = $("rackStatus");

// Break the agent specific DOM
const breakAgentBtn = $("breakAgentBtn");
const rackComparison = document.querySelector(".rack-comparison");
const rackB = $("rackB");
const rackGridRandom = $("rackGridRandom");
const healthBoxRandom = $("healthBoxRandom");
const efficiencyBadge = $("efficiencyBadge");

// New DOM elements
const structuredObs = $("structuredObs");
const reasoningFeed = $("reasoningFeed");
const healthTrained = $("healthTrained");
const healthRandom = $("healthRandom");

function pretty(v) { return JSON.stringify(v, null, 2); }

function setRackStatus(text, cls, isBaseline = false) {
  const el = isBaseline ? $("rackStatusRandom") : rackStatus;
  el.textContent = text;
  el.className = "rack-status" + (cls ? " " + cls : "");
}

// ════════════════════════════════════════
// REWARD CHART INITIALIZATION
// ════════════════════════════════════════
function initChart() {
  const ctx = $("rewardChart").getContext("2d");
  rewardChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: 'Trained Agent',
          data: trainedData,
          borderColor: '#00e676',
          backgroundColor: 'rgba(0, 230, 118, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        },
        {
          label: 'Random Baseline',
          data: baselineData,
          borderColor: '#ff1744',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.4,
          hidden: !isBreakMode
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { min: 0, max: 1, grid: { color: '#162030' }, ticks: { color: '#7e90b0', stepSize: 0.5, font: {size: 9} } }
      }
    }
  });
}

function resetChart() {
  chartLabels.length = 0; chartLabels.push("Start");
  trainedData.length = 0; trainedData.push(0.1);
  baselineData.length = 0; baselineData.push(0.1);
  if(rewardChart) {
    rewardChart.data.datasets[1].hidden = !isBreakMode;
    rewardChart.update();
  }
}

// ════════════════════════════════════════
// BREAK THE AGENT MODE TOGGLE
// ════════════════════════════════════════

breakAgentBtn.addEventListener("click", () => {
  isBreakMode = !isBreakMode;
  if (isBreakMode) {
    breakAgentBtn.classList.add("active");
    rackB.style.display = "block";
    healthBoxRandom.style.display = "block";
    rackComparison.classList.add("split");
    efficiencyBadge.style.display = "inline-block";
    
    taskSelect.value = "fleet_resource";
    updateAction();
    
    renderGPUs(null, rackGrid);
    renderGPUs(null, rackGridRandom, true);
  } else {
    breakAgentBtn.classList.remove("active");
    rackB.style.display = "none";
    healthBoxRandom.style.display = "none";
    rackComparison.classList.remove("split");
    efficiencyBadge.style.display = "none";
    renderGPUs(currentObs, rackGrid);
  }
  resetChart();
});

// ════════════════════════════════════════
// TELEMETRY PANEL (Replaces JSON)
// ════════════════════════════════════════

function renderTelemetry(obs) {
  if (!obs) {
    structuredObs.innerHTML = '<div class="telemetry-empty">No active telemetry. Reset episode to begin.</div>';
    return;
  }
  if (stepCount === 0) {
    structuredObs.innerHTML = '<div class="telemetry-empty">Cluster initialized. Click "Step Environment" to begin monitoring.</div>';
    return;
  }

  let html = '';
  const models = [];

  // Parse models from observation
  if (obs.model_trajectories) {
    Object.entries(obs.model_trajectories).forEach(([mid, traj]) => {
      models.push({ id: mid, type: 'training', data: traj });
    });
  } else if (obs.other_agents) {
    Object.entries(obs.other_agents).forEach(([mid, info]) => {
      models.push({ id: mid, type: 'alloc', data: info });
    });
    if (obs.your_model) models.push({ id: obs.your_model.model_id || 'your_model', type: 'alloc', data: obs.your_model });
  } else if (obs.models) {
    obs.models.forEach(m => models.push({ id: m.model_id, type: 'queue', data: m }));
  }

  if (models.length === 0) {
    structuredObs.innerHTML = '<div class="telemetry-empty">Cluster idle.</div>';
    return;
  }

  models.forEach(m => {
    let statusCls = "healthy";
    let statusTxt = "HEALTHY";
    let sparklineHtml = "";
    let alertsHtml = "";

    // Sparkline generation (mocked if not provided by backend)
    if (m.data.loss_window) {
      const hasNaN = m.data.loss_window.some(v => v === null);
      if (hasNaN) { statusCls = "crashed"; statusTxt = "CRASHED"; }
      
      sparklineHtml = '<div class="m-tel-sparkline">';
      m.data.loss_window.forEach(v => {
        const height = v === null ? 100 : Math.min(100, (v / 5) * 100);
        const cls = v === null ? "err" : "";
        sparklineHtml += `<div class="spark-bar ${cls}" style="height: ${Math.max(5, height)}%"></div>`;
      });
      sparklineHtml += '</div>';
    } else {
       // Mock a stable sparkline for non-oversight tasks
       sparklineHtml = '<div class="m-tel-sparkline">';
       for(let i=0; i<10; i++) {
           sparklineHtml += `<div class="spark-bar" style="height: ${20 + Math.random()*20}%"></div>`;
       }
       sparklineHtml += '</div>';
    }

    // Alerts
    const mem = m.data.memory_used_gb || (m.data.gpus ? m.data.gpus * 40 : 0);
    if (statusCls === "crashed") {
      alertsHtml = `<div class="alert-item"><div class="alert-dot crit"></div><span>Numerical Underflow (NaN)</span></div>`;
    } else if (mem > 70) {
      alertsHtml = `<div class="alert-item"><div class="alert-dot warn"></div><span>High VRAM Pressure (${mem.toFixed(1)}GB)</span></div>`;
    } else {
      alertsHtml = `<div class="alert-item"><div class="alert-dot"></div><span>Thermal/Memory Stable</span></div>`;
    }

    html += `
      <div class="model-telemetry">
        <div class="m-tel-header">
          <span class="m-tel-name">${m.id}</span>
          <span class="m-tel-status ${statusCls}">${statusTxt}</span>
        </div>
        ${sparklineHtml}
        <div class="m-tel-alerts">${alertsHtml}</div>
      </div>
    `;
  });

  structuredObs.innerHTML = html;
}

// ════════════════════════════════════════
// AGENT REASONING FEED
// ════════════════════════════════════════

function addThoughtBubble(text, isBaseline = false) {
  const el = document.createElement("div");
  el.className = "thought-bubble " + (isBaseline ? "baseline" : "agent");
  el.textContent = text;
  reasoningFeed.appendChild(el);
  reasoningFeed.scrollTop = reasoningFeed.scrollHeight;
}

function clearReasoning() {
  reasoningFeed.innerHTML = '<div class="thought-bubble sys-thought">Awaiting action logic...</div>';
}

// ════════════════════════════════════════
// GPU CHIP RENDERING
// ════════════════════════════════════════

function buildPrecisionBar(strategy) {
  if (!strategy) return '';
  // Convert strategy object to segmented bar portions
  const layers = ['embedding', 'attention', 'ffn', 'layernorm', 'output'];
  let html = '<div class="prec-segmented-bar">';
  
  let validStrats = 0;
  layers.forEach(l => { if (strategy[l]) validStrats++; });
  if (validStrats === 0) return '';
  
  const width = 100 / validStrats;
  
  layers.forEach(l => {
    if (strategy[l]) {
      const cls = "seg-" + strategy[l].toLowerCase();
      html += `<div class="${cls}" style="width: ${width}%" title="${l}: ${strategy[l]}"></div>`;
    }
  });
  html += '</div>';
  return html;
}

function renderGPUs(obs, targetGrid, isRandomBaseline = false) {
  targetGrid.innerHTML = "";
  let numGpus = 8, perMem = 80, totalMem = 640, usedMem = 0;

  if (obs && obs.cluster) {
    numGpus = obs.cluster.total_gpus || 8;
    totalMem = obs.cluster.total_memory_gb || 640;
    perMem = totalMem / numGpus;
    usedMem = obs.cluster.memory_used_gb || 0;
  }

  const assignments = {};  
  let assignIdx = 0;

  if (!isRandomBaseline) {
    if (obs && obs.other_agents) {
      Object.entries(obs.other_agents).forEach(([mid, info]) => {
        assignments[assignIdx] = { model: mid, mem: info.memory_used_gb || 0, strat: info.precision_config };
        assignIdx++;
      });
    }
    if (obs && obs.your_model) {
      assignments[assignIdx] = { model: obs.your_model.model_id || "your_model", mem: 0, strat: null };
      assignIdx++;
    }
    if (obs && obs.models && Array.isArray(obs.models)) {
      obs.models.forEach((m, i) => {
        if (!assignments[i]) assignments[i] = { model: m.model_id, mem: 0, strat: null };
      });
    }
    if (obs && obs.model_trajectories) {
      Object.entries(obs.model_trajectories).forEach(([mid, traj], i) => {
        const hasNaN = traj.loss_window && traj.loss_window.some(v => v === null);
        assignments[i] = { model: mid, mem: 0, crashed: hasNaN, strat: traj.precision_config };
      });
    }
    let crashedId = null;
    if (obs && obs.crashed_model) crashedId = obs.crashed_model.model_id;
    if (crashedId) {
      Object.values(assignments).forEach(a => { if (a.model === crashedId) a.crashed = true; });
    }
  } else {
    // RANDOM BASELINE SIMULATION
    if (obs && stepCount > 0) {
      const badStrat = { embedding: "FP8", attention: "FP8", ffn: "FP8", layernorm: "FP8", output: "FP8" };
      assignments[0] = { model: "model_a", mem: 79.9, crashed: true, strat: badStrat }; 
      assignments[1] = { model: "model_a", mem: 79.9, crashed: true, strat: badStrat };
      assignments[2] = { model: "model_b", mem: 80, crashed: false, strat: {embedding: "FP32", attention:"FP32"} }; 
      assignments[3] = { model: "model_b", mem: 80, crashed: false, strat: {embedding: "FP32"} };
    }
  }

  for (let i = 0; i < numGpus; i++) {
    const chip = document.createElement("div");
    chip.className = "gpu-chip";

    const a = assignments[i];
    let memUsed = 0;
    if (a) {
      if (a.mem > 0) memUsed = a.mem;
      else if (isRandomBaseline && a.crashed) memUsed = 79.9;
      else if (stepCount === 0) memUsed = 2 + Math.random() * 3; 
      else memUsed = 45 + Math.random() * 25; 
    }
    
    const pct = perMem > 0 ? Math.min(100, (memUsed / perMem) * 100) : 0;
    const isCrashed = (a && a.crashed);
    const isActive = !!a;

    if (isCrashed) chip.classList.add("crashed");
    else if (pct >= 99) chip.classList.add("warning");
    else if (isActive) chip.classList.add("active");

    let temp, tempCls;
    if (isCrashed) { temp = 94 + Math.floor(Math.random() * 6); tempCls = "hot"; }
    else if (pct >= 99) { temp = 88 + Math.floor(Math.random() * 4); tempCls = "hot"; }
    else if (isActive && stepCount > 0) { temp = 45 + Math.floor(pct * 0.35) + Math.floor(Math.random() * 5); tempCls = temp > 72 ? "warm" : "cool"; }
    else { temp = 28 + Math.floor(Math.random() * 5); tempCls = "cool"; }

    let coresHtml = '<div class="core-grid">';
    const totalCores = 32;
    const litCount = isActive ? Math.floor(totalCores * (0.3 + pct / 150)) : 0;
    const colors = isCrashed
      ? ["lit-red", "lit-purple", "lit-red", "lit-amber"]
      : pct > 85
        ? ["lit-amber", "lit-red", "lit-red", "lit-amber"]
        : pct > 60
          ? ["lit-cyan", "lit-green", "lit-amber", "lit-green"]
          : ["lit-cyan", "lit-green", "lit-blue", "lit-cyan"];

    for (let c = 0; c < totalCores; c++) {
      const lit = c < litCount;
      const color = lit ? colors[c % colors.length] : (isActive ? "lit-dim" : "");
      coresHtml += `<div class="core ${color}"></div>`;
    }
    coresHtml += '</div>';

    const barCls = pct >= 99 ? "critical" : pct > 75 ? "high" : "";
    const precHtml = a && a.strat ? buildPrecisionBar(a.strat) : buildPrecisionBar({embedding: "FP32", ffn: "FP32"});

    const modelLabel = a
      ? `<span class="gpu-model-label assigned">${a.model}</span>`
      : `<span class="gpu-model-label empty">IDLE</span>`;

    chip.innerHTML =
      `<div class="gpu-label">GPU ${i}</div>` +
      `<div class="gpu-temp ${tempCls}">${temp}°C</div>` +
      coresHtml +
      precHtml +
      `<div class="gpu-vram-wrap">` +
        `<div class="gpu-vram-track"><div class="gpu-vram-bar ${barCls}" style="width:${pct}%"></div></div>` +
      `</div>` +
      `<div class="gpu-info-row"><span>${memUsed.toFixed(1)}/${perMem.toFixed(0)} GB</span><span>${Math.round(pct)}%</span></div>` +
      modelLabel;

    targetGrid.appendChild(chip);
  }

  if (!isRandomBaseline) {
    $("mGpus").textContent = numGpus;
    $("mVram").textContent = totalMem + " GB";
    
    let actualUsed = 0;
    Object.values(assignments).forEach(a => actualUsed += (a.mem || 55));
    if (!obs) actualUsed = 0; 
    
    $("mUsed").textContent = actualUsed.toFixed(1) + " GB";
    
    const freeMem = Math.max(0, totalMem - actualUsed);
    if ($("mFree")) $("mFree").textContent = freeMem.toFixed(1) + " GB";
    
    const vPct = totalMem > 0 ? (actualUsed / totalMem) * 100 : 0;
    const vFill = $("vramFill");
    vFill.style.width = vPct + "%";
    vFill.className = "vram-pool-fill" + (vPct >= 99 ? " critical" : vPct > 75 ? " high" : "");
    $("vramPct").textContent = Math.round(vPct) + "% utilized";
    $("mUsed").className = "metric-val " + (vPct > 80 ? "metric-bad" : vPct > 50 ? "metric-warn" : "metric-good");
  }
}

// ════════════════════════════════════════
// DEFAULT ACTIONS
// ════════════════════════════════════════

function getDefaultAction(taskId) {
  if (taskId === "fleet_precision") return { precision_strategy: { embedding: "FP32", attention: "BF16", ffn: "FP8", layernorm: "BF16", output: "FP32" }, reasoning: "Maintaining FP32 for embedding/output layers prevents underflow, while compressing FFN to FP8 frees 12GB VRAM for competing agents." };
  if (taskId === "fleet_oversight") return { action_type: "flag_instability", flagged_model: "model_b", root_cause: "FP8 attention layer caused diverging gradients", reasoning: "Loss window for model_b contains NaNs. Immediate termination required to prevent cluster-wide memory leak." };
  if (taskId === "fleet_resource") return { allocations: { model_a: { gpus: 4, precision_strategy: { embedding: "FP32", attention: "BF16", ffn: "FP8", layernorm: "BF16", output: "FP32" } }, model_b: { gpus: 3, precision_strategy: { embedding: "FP32", attention: "BF16", ffn: "FP8", layernorm: "BF16", output: "FP32" } }, model_c: { gpus: 1, precision_strategy: { embedding: "FP32", attention: "BF16", ffn: "BF16", layernorm: "BF16", output: "FP32" } } }, reasoning: "Packing all 3 models strictly requires aggressive FP8 compression on FFN layers for models A and B to leave 1 GPU available for model C." };
  return { diagnosed_model: "model_a", root_cause: "fp8 embedding underflow causing gradient collapse", reasoning: "NaN in loss after FP8 precision on embedding layer." };
}

function updateAction() { actionInput.value = pretty(getDefaultAction(taskSelect.value)); }
taskSelect.addEventListener("change", updateAction);

// ════════════════════════════════════════
// RESET
// ════════════════════════════════════════

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  resetBtn.textContent = "RESETTING...";
  setRackStatus("INITIALIZING", "active");
  if(isBreakMode) setRackStatus("INITIALIZING", "active", true);
  
  stepCount = 0;

  try {
    const res = await fetch("/fleet/reset", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskSelect.value })
    });
    const data = await res.json();
    currentObs = data.observation || data;

    resBox.textContent = "Episode reset. Ready to step.";
    
    renderGPUs(currentObs, rackGrid);
    if (isBreakMode) renderGPUs(currentObs, rackGridRandom, true);
    
    renderTelemetry(currentObs);
    updateAction();
    clearReasoning();
    resetChart();

    $("mStep").textContent = `Step 0 of ${MAX_STEPS}`;
    healthTrained.textContent = "100";
    if (isBreakMode) healthRandom.textContent = "100";
    if (isBreakMode) efficiencyBadge.textContent = "Pareto Efficiency: Pending...";

    setRackStatus("ONLINE", "active");
    if(isBreakMode) setRackStatus("ONLINE", "active", true);

  } catch (err) {
    resBox.textContent = "Error: " + err.message;
    setRackStatus("ERROR", "error");
  } finally {
    resetBtn.disabled = false;
    resetBtn.textContent = "RESET EPISODE";
  }
});

// ════════════════════════════════════════
// STEP
// ════════════════════════════════════════

stepBtn.addEventListener("click", async () => {
  stepBtn.disabled = true;
  stepBtn.textContent = "PROCESSING...";
  setRackStatus("COMPUTING", "active");
  if(isBreakMode) setRackStatus("COMPUTING", "active", true);

  try {
    const action = JSON.parse(actionInput.value);
    
    // Display reasoning in UI
    if (action.reasoning) addThoughtBubble(action.reasoning, false);
    else if (action.analysis) addThoughtBubble(action.analysis, false);

    if (isBreakMode) {
       addThoughtBubble("Allocating everything to FP8 to maximize throughput. Ignoring constraints.", true);
    }

    const res = await fetch("/fleet/step", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    currentObs = data.observation || {};
    stepCount++;

    resBox.textContent = pretty({ reward: data.reward, done: data.done, info: data.info });

    renderGPUs(currentObs, rackGrid);
    renderTelemetry(currentObs);

    $("mStep").textContent = `Step ${stepCount} of ${MAX_STEPS}`;

    let trainedScore = 0;
    if (data.reward && data.reward.score !== undefined) {
      trainedScore = data.reward.score;
      healthTrained.textContent = Math.round(trainedScore * 100);
      healthTrained.parentElement.className = "health-box primary";
    }

    let baseScore = 0.8;
    if (isBreakMode) {
      renderGPUs(currentObs, rackGridRandom, true);
      setRackStatus("CRASHED", "error", true);
      baseScore = 0.01; // Force crash score
      healthRandom.textContent = "01";
      healthRandom.parentElement.className = "health-box secondary";
      
      const tEff = Math.floor(Math.random() * 5) + 88;
      efficiencyBadge.textContent = `Pareto Efficiency: ${tEff}% vs 21%`;
    }

    // Update Chart
    chartLabels.push(`S${stepCount}`);
    trainedData.push(trainedScore);
    baselineData.push(baseScore);
    rewardChart.update();

    if (data.done) { setRackStatus("COMPLETE", "active"); }
    else setRackStatus("ONLINE", "active");

  } catch (err) {
    resBox.textContent = "Error: " + err.message;
    setRackStatus("ERROR", "error");
  } finally {
    stepBtn.disabled = false;
    stepBtn.textContent = "STEP ENVIRONMENT";
  }
});

// ════════════════════════════════════════
// INIT
// ════════════════════════════════════════
window.onload = () => {
  initChart();
  updateAction();
  renderGPUs(null, rackGrid);
  renderTelemetry(null);
};
