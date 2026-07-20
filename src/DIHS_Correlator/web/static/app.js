function readAppState() {
  const node = document.getElementById("app-state");
  if (!node) {
    return null;
  }

  try {
    return JSON.parse(node.textContent);
  } catch (error) {
    return null;
  }
}

function setupAnalysisForm(state) {
  const classColumnSelect = document.getElementById("class-column");
  const unknownSampleSelect = document.getElementById("unknown-sample");
  const modeSelect = document.getElementById("mode");
  const modelGroup = document.getElementById("model-group");
  const perturbativeSettings = document.getElementById("perturbative-settings");
  const resolvednessSettings = document.getElementById("resolvedness-settings");
  const outputDirInput = document.getElementById("output-dir");

  if (
    !classColumnSelect ||
    !unknownSampleSelect ||
    !modeSelect ||
    !modelGroup ||
    !perturbativeSettings ||
    !resolvednessSettings ||
    !outputDirInput
  ) {
    return;
  }

  function syncUnknownOptions(preferredToken) {
    const options = state.unknownOptions[classColumnSelect.value] || [];
    const currentToken = preferredToken || unknownSampleSelect.value;
    unknownSampleSelect.innerHTML = "";

    options.forEach((option) => {
      const element = document.createElement("option");
      element.value = option.token;
      element.textContent = option.label;
      if (option.token === currentToken) {
        element.selected = true;
      }
      unknownSampleSelect.appendChild(element);
    });

    if (!unknownSampleSelect.value && options.length > 0) {
      unknownSampleSelect.value = options[0].token;
    }
  }

  function syncModeUI(useInitialValue) {
    const mode = modeSelect.value;
    modelGroup.classList.toggle(
      "hidden",
      !state.modelRequiredModes.includes(mode),
    );
    perturbativeSettings.classList.toggle(
      "hidden",
      !state.perturbativeModes.includes(mode),
    );
    resolvednessSettings.classList.toggle(
      "hidden",
      !state.resolvednessModes.includes(mode),
    );

    if (!useInitialValue && state.defaultOutputDirs[mode]) {
      outputDirInput.value = state.defaultOutputDirs[mode];
    }
  }

  classColumnSelect.addEventListener("change", () => syncUnknownOptions(""));
  modeSelect.addEventListener("change", () => syncModeUI(false));

  syncUnknownOptions(state.initialUnknownToken);
  modeSelect.value = state.initialMode;
  syncModeUI(true);
}

function setupJobPolling() {
  const jobPanel = document.getElementById("job-panel");
  const submitButton = document.querySelector(
    "#analysis-form button[type='submit']",
  );

  if (!jobPanel || !submitButton) {
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Run In Progress";

  const jobId = jobPanel.dataset.jobId;
  const datasetId = jobPanel.dataset.datasetId;
  const progressFill = document.getElementById("progress-fill");
  const jobStatus = document.getElementById("job-status");
  const jobProgress = document.getElementById("job-progress");
  const jobStage = document.getElementById("job-stage");
  const jobCounter = document.getElementById("job-counter");
  const jobMessage = document.getElementById("job-message");
  const jobLogs = document.getElementById("job-logs");
  const jobLogBox = document.getElementById("job-log-box");

  async function pollJob() {
    try {
      const response = await fetch(`/jobs/${jobId}/status`, {
        cache: "no-store",
      });
      if (!response.ok) {
        window.setTimeout(pollJob, 1500);
        return;
      }

      const payload = await response.json();
      progressFill.style.width = `${payload.progress_percent}%`;
      jobStatus.textContent = payload.status
        .replaceAll("_", " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
      jobProgress.textContent = `${payload.progress_percent}%`;
      jobStage.textContent = payload.stage || "running";
      jobCounter.textContent =
        payload.current !== null && payload.total !== null
          ? `${payload.current} / ${payload.total}`
          : "Working";
      jobMessage.innerHTML = `<strong>Current task:</strong> ${payload.message || "Working"}`;

      if (payload.logs) {
        jobLogs.textContent = payload.logs;
        jobLogBox.open = true;
      }

      if (payload.status === "completed" && payload.result_id) {
        window.location.href = `/?dataset=${encodeURIComponent(datasetId)}&result=${encodeURIComponent(payload.result_id)}`;
        return;
      }

      if (payload.status === "error") {
        submitButton.disabled = false;
        submitButton.textContent = "Run Analysis";
        return;
      }
    } catch (error) {
      // Keep polling on transient client-side failures.
    }

    window.setTimeout(pollJob, 1200);
  }

  pollJob();
}

function setupPlotViewer() {
  const plotViewer = document.getElementById("plot-viewer");
  if (!plotViewer) {
    return;
  }

  const viewerStage = document.getElementById("viewer-stage");
  const viewerImage = document.getElementById("viewer-image");
  const viewerTitle = document.getElementById("viewer-title");
  const viewerClose = document.getElementById("viewer-close");
  const viewerZoomIn = document.getElementById("viewer-zoom-in");
  const viewerZoomOut = document.getElementById("viewer-zoom-out");
  const viewerReset = document.getElementById("viewer-reset");
  const openPlotButtons = document.querySelectorAll(".js-open-plot");

  let scale = 1;
  let translateX = 0;
  let translateY = 0;
  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragOriginX = 0;
  let dragOriginY = 0;

  function renderViewerTransform() {
    viewerImage.style.transform =
      `translate(${translateX}px, ${translateY}px) ` +
      `translate(-50%, -50%) scale(${scale})`;
  }

  function resetViewer() {
    scale = 1;
    translateX = 0;
    translateY = 0;
    renderViewerTransform();
  }

  function closeViewer() {
    plotViewer.classList.remove("open");
    plotViewer.setAttribute("aria-hidden", "true");
    viewerImage.removeAttribute("src");
  }

  function openViewer(src, label) {
    viewerImage.src = src;
    viewerImage.alt = label;
    viewerTitle.textContent = label || "Plot Viewer";
    resetViewer();
    plotViewer.classList.add("open");
    plotViewer.setAttribute("aria-hidden", "false");
  }

  function zoomBy(delta) {
    scale = Math.min(8, Math.max(0.35, Number((scale + delta).toFixed(2))));
    renderViewerTransform();
  }

  openPlotButtons.forEach((element) => {
    element.addEventListener("click", () => {
      openViewer(element.dataset.plotSrc, element.dataset.plotLabel);
    });
  });

  viewerClose.addEventListener("click", closeViewer);
  viewerZoomIn.addEventListener("click", () => zoomBy(0.2));
  viewerZoomOut.addEventListener("click", () => zoomBy(-0.2));
  viewerReset.addEventListener("click", resetViewer);

  plotViewer.addEventListener("click", (event) => {
    if (event.target === plotViewer) {
      closeViewer();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && plotViewer.classList.contains("open")) {
      closeViewer();
    }
  });

  viewerStage.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      zoomBy(event.deltaY < 0 ? 0.12 : -0.12);
    },
    { passive: false },
  );

  viewerStage.addEventListener("pointerdown", (event) => {
    dragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragOriginX = translateX;
    dragOriginY = translateY;
    viewerStage.classList.add("dragging");
    viewerStage.setPointerCapture(event.pointerId);
  });

  viewerStage.addEventListener("pointermove", (event) => {
    if (!dragging) {
      return;
    }
    translateX = dragOriginX + (event.clientX - dragStartX);
    translateY = dragOriginY + (event.clientY - dragStartY);
    renderViewerTransform();
  });

  function stopDragging(event) {
    if (!dragging) {
      return;
    }
    dragging = false;
    viewerStage.classList.remove("dragging");
    if (event && event.pointerId !== undefined) {
      viewerStage.releasePointerCapture(event.pointerId);
    }
  }

  viewerStage.addEventListener("pointerup", stopDragging);
  viewerStage.addEventListener("pointercancel", stopDragging);
}

document.addEventListener("DOMContentLoaded", () => {
  const appState = readAppState();
  if (appState) {
    setupAnalysisForm(appState);
  }

  setupJobPolling();
  setupPlotViewer();
});
