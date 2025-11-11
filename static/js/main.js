document.addEventListener('DOMContentLoaded', () => {
  // 1) Render Feather icons
  if (window.feather && typeof window.feather.replace === 'function') {
    window.feather.replace();
  }

  // 2) Toggle model param sections on Register Model page
  const typeSelect = document.getElementById('model_type');
  if (typeSelect) {
    const sections = {
      keras_mlp: document.getElementById('keras-params'),
      logistic_regression: document.getElementById('logreg-params'),
      svm: document.getElementById('svm-params'),
      random_forest: document.getElementById('rf-params'),
      knn: document.getElementById('knn-params'),
      naive_bayes: document.getElementById('nb-params'),
    };

    const updateVisibility = () => {
      const val = typeSelect.value;
      Object.keys(sections).forEach(key => {
        if (sections[key]) {
          sections[key].classList.toggle('hidden', key !== val);
        }
      });
    };
    typeSelect.addEventListener('change', updateVisibility);
    updateVisibility();
  }

  // 3) First-start modal behavior
  try {
    const state = window.__FIRST_START__ || { noDatasets: false, noModels: false };
    const dismissed = localStorage.getItem('firstStartDismissed') === '1';
    const shouldShow = (state.noDatasets || state.noModels) && !dismissed;
    const modal = document.getElementById('firstStartModal');
    const dismissBtn = document.getElementById('dismissStart');
    if (modal && shouldShow) {
      modal.style.display = 'flex';
    }
    if (dismissBtn) {
      dismissBtn.addEventListener('click', () => {
        localStorage.setItem('firstStartDismissed', '1');
        if (modal) modal.style.display = 'none';
      });
    }
  } catch (e) {
    // No-op: modal state not available
  }

  // 4) Workspace training workflow
  const startBtn = document.getElementById('startTrainingBtn');
  const stopBtn = document.getElementById('stopTrainingBtn');
  const stateEl = document.getElementById('trainingState');
  const progressBar = document.getElementById('trainProgressBar');
  const messagesEl = document.getElementById('trainMessages');
  const form = document.getElementById('trainForm');
  const datasetSel = document.getElementById('dataset_id');
  const modelsSel = document.getElementById('model_ids');
  const inputs = {}; // perceptron-specific params are configured on Register Model

  let pollTimer = null;

  function setRunningUI(running) {
    if (!startBtn || !stopBtn) return;
    startBtn.disabled = running;
    stopBtn.disabled = !running;
    if (datasetSel) datasetSel.disabled = running;
    if (modelsSel) modelsSel.disabled = running;
    if (stateEl) stateEl.textContent = running ? 'Training in progress…' : '';
  }

  function setProgress(p) {
    const pct = Math.max(0, Math.min(100, Math.round(p * 100)));
    if (progressBar) progressBar.style.width = pct + '%';
  }

  function showMessages(msgs) {
    if (!messagesEl) return;
    messagesEl.textContent = Array.isArray(msgs) ? msgs.join(' • ') : '';
  }

  async function pollStatus(onDone) {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();
      if (typeof data.progress === 'number') setProgress(data.progress);
      showMessages(data.messages || []);
      if (data.status === 'completed') {
        clearInterval(pollTimer); pollTimer = null;
        setRunningUI(false);
        if (stateEl) stateEl.textContent = 'Training completed.';
        // Reload to reflect new metrics/charts, preserving selected dataset
        setTimeout(() => {
          const ds = datasetSel && datasetSel.value;
          if (ds) {
            window.location.href = '/workspace?dataset_id=' + encodeURIComponent(ds);
          } else {
            window.location.reload();
          }
        }, 500);
      } else if (data.status === 'error') {
        clearInterval(pollTimer); pollTimer = null;
        setRunningUI(false);
        if (stateEl) stateEl.textContent = 'Error: ' + (data.error || 'Training failed');
      }
    } catch (e) {
      // If polling fails momentarily, keep trying; show a subtle hint
      if (stateEl) stateEl.textContent = 'Updating status…';
    }
  }

  async function startTraining() {
    if (!datasetSel || !modelsSel) return;
    const dataset_id = datasetSel.value;
    const model_ids = Array.from(modelsSel.selectedOptions).map(o => o.value);
    if (!dataset_id || model_ids.length === 0) {
      if (stateEl) stateEl.textContent = 'Select a dataset and at least one model.';
      return;
    }
    // Validate numeric inputs
    // No extra validation on epochs/optimizer/loss here; perceptron params live in Register Model

    setRunningUI(true);
    setProgress(0);
    showMessages([]);

    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id, model_ids })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        if (data && data.error === 'job_running') {
          if (stateEl) stateEl.textContent = 'A training job is already running.';
        } else if (data && data.error === 'invalid_request') {
          if (stateEl) stateEl.textContent = 'Invalid request. Check selections.';
        } else {
          if (stateEl) stateEl.textContent = 'Failed to start training.';
        }
        setRunningUI(false);
        return;
      }
      // Begin polling status
      pollTimer = setInterval(() => { pollStatus(); }, 1000);
      // Immediate status check for responsiveness
      pollStatus();
    } catch (e) {
      setRunningUI(false);
      if (stateEl) stateEl.textContent = 'Network error starting training.';
    }
  }

  async function stopTraining() {
    try {
      if (stateEl) stateEl.textContent = 'Stopping…';
      await fetch('/api/train/stop', { method: 'POST' });
    } catch (e) {
      if (stateEl) stateEl.textContent = 'Stop request failed.';
    }
  }

  if (startBtn) startBtn.addEventListener('click', startTraining);
  if (stopBtn) stopBtn.addEventListener('click', stopTraining);

  // Show epoch progress if available
  const updateEpochInState = async () => {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();
      if (data.status === 'running' && typeof data.epoch_total === 'number' && data.epoch_total > 0) {
        if (stateEl) stateEl.textContent = `Training in progress… (epoch ${data.epoch}/${data.epoch_total})`;
      }
    } catch {}
  };
  if (startBtn) setInterval(updateEpochInState, 1000);
});