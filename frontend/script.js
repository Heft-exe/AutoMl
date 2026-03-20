/**
 * script.js — AutoML Frontend Logic
 * Handles: file upload, training, results display, prediction form
 */

const API = "http://127.0.0.1:8000";

// ── App State ─────────────────────────────────────────────────────────────────
let appState = {
  columns: [],
  taskType: null,
  featureColumns: [],
  trained: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const dataSection    = document.getElementById("dataSection");
const trainSection   = document.getElementById("trainSection");
const resultsSection = document.getElementById("resultsSection");
const predictSection = document.getElementById("predictSection");

// ── Drag & Drop ───────────────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFileUpload(fileInput.files[0]); });

// ── Step 1: Upload ────────────────────────────────────────────────────────────
async function handleFileUpload(file) {
  if (!file.name.endsWith(".csv")) { showToast("Please upload a CSV file.", "error"); return; }

  showLoading("uploadStatus", "Uploading & analysing your dataset…");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch(`${API}/upload/`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    appState.columns = data.columns;
    renderPreview(data);

    const select = document.getElementById("targetSelect");
    select.innerHTML = data.columns.map((c) => `<option value="${c}">${c}</option>`).join("");

    hideLoading("uploadStatus");
    show(dataSection);
    show(trainSection);
    dropzone.querySelector(".drop-icon").textContent = "✅";
    dropzone.querySelector(".drop-text").textContent = `${file.name} loaded`;
    showToast("Dataset uploaded successfully!", "success");
  } catch (err) {
    hideLoading("uploadStatus");
    showToast(err.message, "error");
  }
}

function renderPreview(data) {
  document.getElementById("statRows").textContent    = data.shape[0].toLocaleString();
  document.getElementById("statCols").textContent    = data.shape[1];
  const missing = Object.values(data.missing_values).reduce((a, b) => a + b, 0);
  document.getElementById("statMissing").textContent = missing;

  document.getElementById("previewHead").innerHTML =
    `<tr>${data.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  document.getElementById("previewBody").innerHTML = data.preview
    .map((row) => `<tr>${data.columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`)
    .join("");
}

// ── Step 2: Train ─────────────────────────────────────────────────────────────
async function triggerTraining() {
  const target = document.getElementById("targetSelect").value;
  if (!target) { showToast("Please select a target column.", "error"); return; }

  const btn = document.getElementById("trainBtn");
  btn.disabled = true;
  btn.textContent = "Training…";
  showLoading("trainStatus", "Running AutoML — comparing models, please wait…");
  hide(resultsSection); hide(predictSection);

  try {
    const res  = await fetch(`${API}/train/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column: target }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Training failed");

    appState.taskType       = data.task_type;
    appState.featureColumns = appState.columns.filter((c) => c !== target);
    appState.trained        = true;

    hideLoading("trainStatus");
    showResults(data, target);
    buildPredictForm();
    show(resultsSection);
    show(predictSection);
    showToast(`Best model: ${data.best_model_name}`, "success");
  } catch (err) {
    hideLoading("trainStatus");
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Train Model";
  }
}

// ── Step 3: Results ───────────────────────────────────────────────────────────
function showResults(data, target) {
  document.getElementById("bestModelName").textContent = data.best_model_name;
  document.getElementById("taskTypeBadge").textContent =
    data.task_type === "classification" ? "Classification" : "Regression";
  document.getElementById("targetBadge").textContent = `Target: ${target}`;

  // Metrics
  const priorityKeys = ["Accuracy","AUC","Recall","Prec.","F1","Kappa","MAE","MSE","RMSE","R2","MAPE"];
  const numericKeys  = Object.keys(data.metrics).filter(
    (k) => k !== "Model" && typeof data.metrics[k] === "number"
  );
  const displayKeys = priorityKeys.filter((k) => numericKeys.includes(k))
    .concat(numericKeys.filter((k) => !priorityKeys.includes(k)));

  document.getElementById("metricsGrid").innerHTML = displayKeys
    .map((k) => `
      <div class="metric-card">
        <div class="metric-label">${k}</div>
        <div class="metric-value">${data.metrics[k]}</div>
      </div>`).join("");

  renderFeatureImportance(data.feature_importance);
  renderComparisonTable(data.comparison_table);
}

function renderFeatureImportance(features) {
  const container = document.getElementById("featureChart");
  if (!features || features.length === 0) {
    container.innerHTML = "<p class='muted'>Feature importance not available for this model type.</p>";
    return;
  }
  const max = features[0].importance || 1;
  container.innerHTML = features.slice(0, 12).map(({ feature, importance }) => `
    <div class="bar-row">
      <span class="bar-label">${feature}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(importance / max) * 100}%"></div>
      </div>
      <span class="bar-value">${importance}</span>
    </div>`).join("");
}

function renderComparisonTable(rows) {
  if (!rows || rows.length === 0) return;
  const keys = Object.keys(rows[0]);
  document.getElementById("comparisonHead").innerHTML =
    `<tr>${keys.map((k) => `<th>${k}</th>`).join("")}</tr>`;
  document.getElementById("comparisonBody").innerHTML = rows.map((row, i) =>
    `<tr class="${i === 0 ? "best-row" : ""}">
      ${keys.map((k) => `<td>${typeof row[k] === "number" ? row[k].toFixed(4) : row[k]}</td>`).join("")}
    </tr>`).join("");
}

// ── Step 4: Predict ───────────────────────────────────────────────────────────
function buildPredictForm() {
  document.getElementById("predictForm").innerHTML = appState.featureColumns.map((col) => `
    <div class="form-group">
      <label for="feat_${col}">${col}</label>
      <input id="feat_${col}" class="form-input" type="text" placeholder="Enter value for ${col}" />
    </div>`).join("");
}

async function sendPredictionRequest() {
  const features = {};
  for (const col of appState.featureColumns) {
    const val = document.getElementById(`feat_${col}`).value.trim();
    features[col] = val !== "" && !isNaN(val) ? parseFloat(val) : val;
  }
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const res  = await fetch(`${API}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");
    displayPrediction(data);
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict";
  }
}

function displayPrediction(data) {
  const box = document.getElementById("predictionResult");
  box.innerHTML = `
    <div class="prediction-output">
      <div class="pred-label">Prediction Result</div>
      <div class="pred-value">${data.prediction}</div>
      ${data.confidence != null
        ? `<div class="pred-confidence">Confidence <strong>${(data.confidence * 100).toFixed(1)}%</strong></div>`
        : ""}
    </div>`;
  show(box);
}

// ── Download ──────────────────────────────────────────────────────────────────
function downloadModel() { window.location.href = `${API}/download/`; }

// ── Utilities ─────────────────────────────────────────────────────────────────
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function showLoading(id, msg) {
  const el = document.getElementById(id);
  el.innerHTML = `<div class="loader-wrap"><span class="spinner"></span><span>${msg}</span></div>`;
  el.classList.remove("hidden");
}
function hideLoading(id) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  el.classList.add("hidden");
}
function showToast(msg, type = "info") {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.className = `toast toast-${type} show`;
  setTimeout(() => toast.classList.remove("show"), 3500);
}
