/**
 * XSIVE — Spectral Curation Engine · app.js
 * Frontend: conecta la UI con la Flask API en /api/*
 */

const API = "http://localhost:5000";

// ── State ──────────────────────────────────────────────────────────
const state = {
  currentTab: "analyze",
  modelLoaded: false,
  trainingActive: false,
  metricsLoaded: false,
};

// ── DOM refs ───────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ══════════════════════════════════════════════════════════════════
// PARTICLES BACKGROUND
// ══════════════════════════════════════════════════════════════════
(function initParticles() {
  const canvas = $("particles-canvas");
  const ctx = canvas.getContext("2d");
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.4 + 0.3,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      alpha: Math.random() * 0.5 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,58,237,${p.alpha})`;
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
    }

    // Lines between close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(124,58,237,${0.08 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => { resize(); createParticles(); });
  resize(); createParticles(); draw();
})();

// ══════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════════════════════════════════
document.querySelectorAll(".nav-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${tab}`).classList.add("active");
    state.currentTab = tab;
    if (tab === "metrics" && !state.metricsLoaded) loadMetrics();
  });
});

// ══════════════════════════════════════════════════════════════════
// STATUS POLLING
// ══════════════════════════════════════════════════════════════════
async function pollStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    updateStatusPill(data);
    state.modelLoaded = data.model_loaded;
    state.trainingActive = data.training_active;
  } catch {
    setStatusPill("error", "Servidor desconectado");
  }
}

function setStatusPill(type, text) {
  const dot  = $("status-dot");
  const span = $("status-text");
  dot.className = `status-dot ${type}`;
  span.textContent = text;
}

function updateStatusPill(data) {
  if (!data.classifier_ready) {
    setStatusPill("warning", "Modo Demo");
  } else if (data.model_loaded) {
    setStatusPill("online", "Modelo Listo");
  } else if (data.model_exists) {
    setStatusPill("warning", "Modelo sin cargar");
  } else {
    setStatusPill("warning", "Sin entrenar");
  }
}

pollStatus();
setInterval(pollStatus, 8000);

// ══════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════
let toastTimer = null;
function showToast(msg, type = "info") {
  const toast = $("toast");
  const icons = { info: "ℹ️", success: "✅", error: "❌", warning: "⚠️" };
  $("toast-icon").textContent = icons[type] || "ℹ️";
  $("toast-msg").textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 4000);
}

// ══════════════════════════════════════════════════════════════════
// UPLOAD & ANALYZE TAB
// ══════════════════════════════════════════════════════════════════
const uploadZone  = $("upload-zone");
const fileInput   = $("audio-file-input");
const processingEl = $("processing-state");
const resultPanel  = $("result-panel");

// Click on upload zone → trigger file input
uploadZone.addEventListener("click", (e) => {
  if (e.target.closest("label")) return;
  fileInput.click();
});

// Drag & Drop
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) analyzeFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) analyzeFile(fileInput.files[0]);
});

$("btn-reset").addEventListener("click", resetToUpload);

function resetToUpload() {
  uploadZone.classList.remove("hidden");
  processingEl.classList.add("hidden");
  resultPanel.classList.add("hidden");
  fileInput.value = "";
  // Reset steps
  ["step-load","step-mel","step-mfcc","step-rms","step-cnn"].forEach((id) => {
    const el = $(id);
    el.classList.remove("active","done");
  });
}

async function analyzeFile(file) {
  const SUPPORTED = [".wav",".mp3",".flac",".aif",".aiff",".ogg",".m4a"];
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!SUPPORTED.includes(ext)) {
    showToast(`Formato no soportado: ${ext}`, "error");
    return;
  }

  // Show processing state
  uploadZone.classList.add("hidden");
  resultPanel.classList.add("hidden");
  processingEl.classList.remove("hidden");

  // Animate steps sequentially
  const steps = ["step-load","step-mel","step-mfcc","step-rms","step-cnn"];
  const labels = [
    "Cargando audio · 22050 Hz...",
    "Generando espectrograma de Mel (128 bins)...",
    "Extrayendo MFCCs + Δ + Δ² (60 coef.)...",
    "Calculando energía RMS (dinámica)...",
    "Ejecutando inferencia CNN...",
  ];

  // Build FormData
  const form = new FormData();
  form.append("file", file);

  // Start fake step animation
  let stepIdx = 0;
  function advanceStep() {
    if (stepIdx > 0) {
      const prev = $(steps[stepIdx - 1]);
      prev.classList.remove("active");
      prev.classList.add("done");
    }
    if (stepIdx < steps.length) {
      $(steps[stepIdx]).classList.add("active");
      $("processing-label").textContent = labels[stepIdx];
      stepIdx++;
    }
  }
  advanceStep();

  // Stagger the step animations
  const stepIntervals = [0, 500, 1000, 1500, 2000];
  stepIntervals.forEach((ms, i) => {
    setTimeout(() => { if (i > 0 && i < steps.length) advanceStep(); }, ms);
  });

  try {
    const res = await fetch(`${API}/api/predict`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Error en el servidor");
    }
    const data = await res.json();

    // Mark all steps done
    steps.forEach((id) => { $(id).classList.remove("active"); $(id).classList.add("done"); });
    await sleep(400);

    showResults(data);
  } catch (err) {
    processingEl.classList.add("hidden");
    uploadZone.classList.remove("hidden");
    showToast(`Error: ${err.message}`, "error");
  }
}

function showResults(data) {
  processingEl.classList.add("hidden");
  resultPanel.classList.remove("hidden");

  const isApproved = data.verdict === "APROBADO";
  const prob = data.probability;
  const pct  = (prob * 100).toFixed(1);

  // Verdict badge
  const card  = $("verdict-card");
  const badge = $("verdict-badge");
  const icon  = $("badge-icon");

  card.classList.remove("approved","denied");
  card.classList.add(isApproved ? "approved" : "denied");
  icon.classList.remove("approved","denied");
  icon.classList.add(isApproved ? "approved" : "denied");
  icon.textContent = isApproved ? "✅" : "🚫";

  const labelEl = $("verdict-label");
  labelEl.textContent = data.verdict;
  labelEl.className = `verdict-label ${isApproved ? "approved" : "denied"}`;
  $("verdict-class").textContent = data.label;

  // Probability bar (animate after short delay)
  $("prob-value").textContent = `${pct}%`;
  setTimeout(() => {
    $("prob-bar-fill").style.width = `${pct}%`;
  }, 100);

  // Meta chips
  const confColors = { ALTA: "#10B981", MEDIA: "#F59E0B", BAJA: "#EF4444" };
  $("confidence-text").textContent = data.confidence;
  $("chip-confidence").style.borderColor = confColors[data.confidence] || "transparent";
  $("time-text").textContent = `${data.processing_ms} ms`;
  $("model-text").textContent = data.model_used ? "CNN Entrenada" : "Demo";

  // Spectrogram
  if (data.spectrogram) {
    $("spectrogram-img").src = data.spectrogram;
    $("spec-filename").textContent = data.filename;
  }

  // Animate card entrance
  card.style.opacity = "0"; card.style.transform = "translateY(20px)";
  requestAnimationFrame(() => {
    card.style.transition = "opacity .5s ease, transform .5s ease";
    card.style.opacity = "1"; card.style.transform = "none";
  });
}

// ══════════════════════════════════════════════════════════════════
// METRICS TAB
// ══════════════════════════════════════════════════════════════════
$("btn-refresh-metrics").addEventListener("click", loadMetrics);

async function loadMetrics() {
  try {
    const res  = await fetch(`${API}/api/metrics`);
    const data = await res.json();
    state.metricsLoaded = true;

    // Parse history CSV for summary metrics
    if (data.history && data.history.length > 0) {
      const last = data.history[data.history.length - 1];
      animateMetric("metric-accuracy", last.val_accuracy, true);
      animateMetric("metric-auc",      last.val_auc,      false);
      animateMetric("metric-precision", last.val_precision || null, false);
      animateMetric("metric-recall",    last.val_recall    || null, false);
    }

    // Load images
    const imgMap = {
      "confusion_matrix":   ["cm-placeholder",     "chart-cm"],
      "roc_pr":             ["roc-placeholder",    "chart-roc"],
      "training_curves":    ["curves-placeholder", "chart-curves"],
    };

    for (const [key, [placeholderId, cardId]] of Object.entries(imgMap)) {
      if (data.images && data.images[key]) {
        const card = $(cardId);
        const body = card.querySelector(".chart-body");
        body.innerHTML = `<img src="${data.images[key]}" alt="${key}" />`;
      }
    }

    if (!data.has_metrics && (!data.history || data.history.length === 0)) {
      showToast("Sin métricas disponibles. Entrena el modelo primero.", "warning");
    } else {
      showToast("Métricas actualizadas", "success");
    }
  } catch {
    showToast("Error al cargar métricas", "error");
  }
}

function animateMetric(cardId, value, isPercent) {
  const card = $(cardId);
  if (!card || value == null) return;
  const span = card.querySelector(".metric-value");
  const target = parseFloat(value);
  const display = isPercent ? (target * 100).toFixed(1) + "%" : target.toFixed(4);
  span.textContent = display;
  span.style.animation = "none";
  requestAnimationFrame(() => {
    span.style.animation = "fadeIn .5s ease";
  });
}

// ══════════════════════════════════════════════════════════════════
// TRAIN TAB
// ══════════════════════════════════════════════════════════════════
$("btn-start-train").addEventListener("click", startTraining);
$("btn-clear-log").addEventListener("click", () => {
  $("train-log-body").innerHTML = '<span class="log-placeholder">Log limpiado...</span>';
});

async function startTraining() {
  if (state.trainingActive) {
    showToast("Ya hay un entrenamiento en curso.", "warning");
    return;
  }

  const fromProcessed = $("from-processed-toggle").checked;
  const logBody = $("train-log-body");
  logBody.innerHTML = "";
  $("log-dot").classList.add("active");
  $("btn-start-train").disabled = true;
  $("btn-start-train").style.opacity = "0.5";

  try {
    const res = await fetch(`${API}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_processed: fromProcessed }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "No se pudo iniciar el entrenamiento");
    }

    showToast("¡Entrenamiento iniciado!", "success");
    subscribeToTrainProgress();
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    $("log-dot").classList.remove("active");
    $("btn-start-train").disabled = false;
    $("btn-start-train").style.opacity = "1";
  }
}

function subscribeToTrainProgress() {
  const logBody  = $("train-log-body");
  const evtSrc   = new EventSource(`${API}/api/train/progress`);

  evtSrc.onmessage = (e) => {
    const entry = JSON.parse(e.data);

    if (entry.msg === "__DONE__") {
      evtSrc.close();
      $("log-dot").classList.remove("active");
      $("btn-start-train").disabled = false;
      $("btn-start-train").style.opacity = "1";
      appendLog(logBody, "──── Entrenamiento finalizado ────", "success");
      pollStatus();
      return;
    }

    const isSuccess = entry.msg.startsWith("✅") || entry.msg.startsWith("✓");
    const isError   = entry.msg.startsWith("❌");
    appendLog(logBody, entry.msg, isSuccess ? "success" : isError ? "error" : "");
  };

  evtSrc.onerror = () => {
    evtSrc.close();
    $("log-dot").classList.remove("active");
    $("btn-start-train").disabled = false;
    $("btn-start-train").style.opacity = "1";
    appendLog(logBody, "⚠ Conexión al servidor perdida.", "error");
  };
}

function appendLog(container, msg, type = "") {
  const isPlaceholder = container.querySelector(".log-placeholder");
  if (isPlaceholder) isPlaceholder.remove();

  const line = document.createElement("div");
  line.className = "log-line";
  const now = new Date().toLocaleTimeString("es-MX", { hour12: false });
  line.innerHTML = `
    <span class="log-time">${now}</span>
    <span class="log-msg ${type}">${escapeHtml(msg)}</span>
  `;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

// ══════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
