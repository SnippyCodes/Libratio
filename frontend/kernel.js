/* ═══════════════════════════════════════════════════
   Agentic Kernel — Live Performance Dashboard
   
   This script powers the interactive Agentic Kernel panel.
   It hits the /kernel/benchmark and /kernel/profile endpoints
   to get real timing data from the physics engine, then
   renders it with animated numbers and comparison bars.
   ═══════════════════════════════════════════════════ */

let kernelPanelOpen = false;

function toggleKernelPanel() {
  const panel = document.getElementById("kernelPanel");
  const badge = document.getElementById("kernelToggle");
  kernelPanelOpen = !kernelPanelOpen;
  
  if (kernelPanelOpen) {
    panel.style.display = "block";
    badge.classList.add("active");
    // Auto-resize the dashboard below
    document.querySelector(".dashboard").style.height = "calc(100vh - 44px - " + panel.offsetHeight + "px)";
    
    // Auto-run profile on first open
    runKernelProfile();
  } else {
    panel.style.display = "none";
    badge.classList.remove("active");
    document.querySelector(".dashboard").style.height = "calc(100vh - 44px)";
  }
}

// Animated number counter
function animateNumber(elementId, targetValue, suffix, duration) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const startValue = 0;
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startValue + (targetValue - startValue) * eased);
    el.textContent = current.toLocaleString() + (suffix || "");
    
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

async function runKernelBenchmark() {
  const btn = document.getElementById("runBenchBtn");
  btn.disabled = true;
  btn.textContent = "BENCHMARKING... (evaluating 1,000 trajectories)";
  btn.classList.add("running");
  
  // Reset values
  document.getElementById("kThroughput").textContent = "...";
  document.getElementById("kLatency").textContent = "...";
  document.getElementById("kSpeedup").textContent = "...";
  document.getElementById("kStatus").textContent = "RUNNING";
  document.getElementById("kStatus").className = "k-big-number";

  try {
    const res = await fetch("/kernel/benchmark");
    const data = await res.json();
    
    // Animate the big numbers
    animateNumber("kThroughput", data.throughput.trajectories_per_second, "", 1200);
    
    const latency = data.timing.total_per_trajectory_us;
    document.getElementById("kLatency").textContent = latency.toFixed(1);
    
    animateNumber("kSpeedup", data.throughput.speedup_vs_docker, "x", 1000);
    
    // Sandbox ratio status
    const statusEl = document.getElementById("kStatus");
    if (data.comparison.sandbox_ratio_solved) {
      statusEl.textContent = "SOLVED";
      statusEl.className = "k-big-number k-solved";
      document.getElementById("kStatusUnit").textContent = "environment is not the bottleneck";
      document.getElementById("kStatusSub").textContent = latency.toFixed(1) + " us << 150,000 us (Docker)";
    } else {
      statusEl.textContent = "ACTIVE";
      statusEl.className = "k-big-number k-active";
    }
    
    // Update subtexts
    document.getElementById("kThroughputSub").textContent = 
      "1,000 trajectories evaluated in " + (data.timing.cost_model_total_us / 1000).toFixed(1) + " ms";
    document.getElementById("kLatencySub").textContent = 
      "Cost: " + data.timing.cost_model_per_eval_us + " us | Safety: " + data.timing.safety_check_per_eval_us + " us | Scoring: " + data.timing.layer_scoring_per_eval_us + " us";
    document.getElementById("kSpeedupSub").textContent = 
      data.throughput.speedup_vs_microvm + "x faster than Firecracker MicroVM";
    
    // Update module bars
    const totalUs = data.timing.cost_model_per_eval_us + data.timing.safety_check_per_eval_us + data.timing.layer_scoring_per_eval_us;
    
    document.getElementById("kmBar1").style.width = (data.timing.cost_model_per_eval_us / totalUs * 100) + "%";
    document.getElementById("kmVal1").textContent = data.timing.cost_model_per_eval_us + " us/eval";
    
    document.getElementById("kmBar2").style.width = (data.timing.safety_check_per_eval_us / totalUs * 100) + "%";
    document.getElementById("kmVal2").textContent = data.timing.safety_check_per_eval_us + " us/eval";
    
    document.getElementById("kmBar3").style.width = (data.timing.layer_scoring_per_eval_us / totalUs * 100) + "%";
    document.getElementById("kmVal3").textContent = data.timing.layer_scoring_per_eval_us + " us/eval";
    
    // Update comparison bar (kernel bar is tiny compared to Docker/VM)
    const kernelPct = Math.max(1, (latency / 150000) * 100);
    document.getElementById("kcBarKernel").style.width = kernelPct + "%";
    document.getElementById("kcValKernel").textContent = latency.toFixed(1) + " us";
    
  } catch (err) {
    document.getElementById("kStatus").textContent = "ERROR";
    document.getElementById("kStatus").className = "k-big-number k-error";
    document.getElementById("kStatusSub").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "RUN BENCHMARK (1,000 trajectories)";
    btn.classList.remove("running");
  }
}

async function runKernelProfile() {
  try {
    const res = await fetch("/kernel/profile");
    const data = await res.json();
    
    if (data.modules) {
      const mods = data.modules;
      const total = data.total_kernel_latency_us || 1;
      
      if (mods.precision_kernel) {
        document.getElementById("kmVal1").textContent = mods.precision_kernel.latency_us + " us";
        document.getElementById("kmBar1").style.width = (mods.precision_kernel.latency_us / total * 100) + "%";
      }
      if (mods.safety_kernel) {
        document.getElementById("kmVal2").textContent = mods.safety_kernel.latency_us + " us";
        document.getElementById("kmBar2").style.width = (mods.safety_kernel.latency_us / total * 100) + "%";
      }
      if (mods.scoring_kernel) {
        document.getElementById("kmVal3").textContent = mods.scoring_kernel.latency_us + " us";
        document.getElementById("kmBar3").style.width = (mods.scoring_kernel.latency_us / total * 100) + "%";
      }
      if (mods.network_kernel) {
        document.getElementById("kmVal4").textContent = mods.network_kernel.latency_us + " us";
        document.getElementById("kmBar4").style.width = (mods.network_kernel.latency_us / total * 100) + "%";
      }
    }
  } catch (e) {
    // Silent fail — profile is supplementary
    console.warn("Kernel profile failed:", e);
  }
}
