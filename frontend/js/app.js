(() => {
  "use strict";

  const API_BASE = ""; // same-origin: served by the FastAPI app itself

  const CLASS_ICONS = {
    airplane: "✈️",
    automobile: "🚗",
    bird: "🐦",
    cat: "🐱",
    deer: "🦌",
    dog: "🐶",
    frog: "🐸",
    horse: "🐴",
    ship: "🚢",
    truck: "🚚",
  };

  const RING_CIRCUMFERENCE = 2 * Math.PI * 42; // r=42, matches the SVG in index.html

  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const errorMsg = document.getElementById("errorMsg");
  const resultPanel = document.getElementById("resultPanel");
  const loadingPanel = document.getElementById("loadingPanel");
  const healthPill = document.getElementById("healthPill");

  const predictedIcon = document.getElementById("predictedIcon");
  const predictedClass = document.getElementById("predictedClass");
  const ringProgress = document.getElementById("ringProgress");
  const confidenceValue = document.getElementById("confidenceValue");
  const explanationText = document.getElementById("explanationText");
  const probList = document.getElementById("probList");
  const resetBtn = document.getElementById("resetBtn");

  let selectedFile = null;

  // ---- Health check ----------------------------------------------------

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("unhealthy");
      healthPill.textContent = "API online";
      healthPill.className = "health-pill ok";
    } catch (err) {
      healthPill.textContent = "API unreachable";
      healthPill.className = "health-pill error";
    }
  }

  // ---- File selection ----------------------------------------------------

  function setSelectedFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showError("Please choose an image file (PNG or JPG).");
      return;
    }
    hideError();
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
    dropzoneEmpty.hidden = true;
    analyzeBtn.disabled = false;
    resultPanel.hidden = true;
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    setSelectedFile(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    setSelectedFile(file);
  });

  function resetState() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewImg.hidden = true;
    dropzoneEmpty.hidden = false;
    analyzeBtn.disabled = true;
    resultPanel.hidden = true;
    hideError();
  }

  resetBtn.addEventListener("click", () => {
    resetState();
    fileInput.click();
  });

  // ---- Analyze -----------------------------------------------------------

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();
    resultPanel.hidden = true;
    loadingPanel.hidden = false;
    analyzeBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      renderResult(data);
    } catch (err) {
      showError(err.message || "Something went wrong while contacting the API.");
    } finally {
      loadingPanel.hidden = true;
      analyzeBtn.disabled = false;
    }
  });

  // ---- Rendering -----------------------------------------------------------

  function renderResult(data) {
    const { class_name: topClass, confidence, probabilities } = data;

    predictedIcon.textContent = CLASS_ICONS[topClass] || "❓";
    predictedClass.textContent = topClass;

    const pct = confidence * 100;
    confidenceValue.textContent = `${pct.toFixed(0)}%`;
    const offset = RING_CIRCUMFERENCE * (1 - confidence);
    // Force reflow so the transition always animates, even on repeated predictions.
    ringProgress.style.transition = "none";
    ringProgress.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    void ringProgress.getBoundingClientRect();
    ringProgress.style.transition = "";
    requestAnimationFrame(() => {
      ringProgress.style.strokeDashoffset = String(offset);
    });

    explanationText.textContent = buildExplanation(probabilities);

    probList.innerHTML = "";
    probabilities.forEach((entry, idx) => {
      const li = document.createElement("li");
      li.className = "prob-row" + (idx === 0 ? " is-top" : "");

      const name = document.createElement("span");
      name.className = "prob-name";
      name.textContent = entry.class_name;

      const track = document.createElement("span");
      track.className = "prob-bar-track";
      const fill = document.createElement("span");
      fill.className = "prob-bar-fill";
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "prob-value";
      value.textContent = `${(entry.probability * 100).toFixed(1)}%`;

      li.append(name, track, value);
      probList.appendChild(li);

      requestAnimationFrame(() => {
        fill.style.width = `${(entry.probability * 100).toFixed(1)}%`;
      });
    });

    resultPanel.hidden = false;
  }

  function buildExplanation(probabilities) {
    const [top, second] = probabilities;
    const topPct = top.probability * 100;
    const secondPct = second ? second.probability * 100 : 0;
    const gapPct = topPct - secondPct;

    let confidencePhrase;
    if (topPct >= 90) confidencePhrase = "very confident";
    else if (topPct >= 70) confidencePhrase = "fairly confident";
    else if (topPct >= 40) confidencePhrase = "not very confident";
    else confidencePhrase = "highly uncertain";

    let gapPhrase;
    if (gapPct >= 40) {
      gapPhrase = `a clear margin over its next guess, ${second.class_name} (${secondPct.toFixed(1)}%)`;
    } else if (gapPct >= 10) {
      gapPhrase = `a moderate margin over its next guess, ${second.class_name} (${secondPct.toFixed(1)}%)`;
    } else {
      gapPhrase = `only a narrow margin over its next guess, ${second.class_name} (${secondPct.toFixed(1)}%) — the two classes look visually similar to the model`;
    }

    return (
      `The model is ${confidencePhrase} (${topPct.toFixed(1)}%) that this is a ${top.class_name}, with ` +
      `${gapPhrase}. This student network never saw a "correct answer" label distribution directly — during ` +
      `training it was taught to reproduce a larger teacher model's full probability spread across all 10 ` +
      `classes, not just the right one, so a narrow gap here usually means the teacher itself saw overlapping ` +
      `visual cues (shape, color, texture) between the two classes.`
    );
  }

  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.hidden = false;
  }

  function hideError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  checkHealth();
})();
