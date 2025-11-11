# AI Model Comparison

A simple, friendly Flask web application to upload datasets, register multiple models (classic ML and a Keras perceptron), train and compare them, and run quick predictions — all in one workspace.

## Features
- Dataset management: upload CSVs, map features and target.
- Model registry: create, edit, delete models with hyperparameters.
- Supported models: `logistic_regression`, `svm`, `random_forest`, `knn`, `naive_bayes`, and `keras_mlp` (perceptron).
- Unified Workspace:
  - Start async training jobs with progress.
  - View latest results per model for the selected dataset.
  - Confusion matrix visualization and summary metrics.
  - Inline prediction UI with selected trained model.
- Perceptron parameters: optimizer and loss via dropdowns; epochs and batch size honored.
- Parameter logging: resolved training parameters are stored per run and shown on result cards.
- Model and training deletion: cascades remove associated files, updates `config.json`.
- Sensible `.gitignore` to keep data and generated artifacts out of version control.

## Quick Start

### Requirements
- Python `3.10+` (Windows/macOS/Linux).
- For perceptron (Keras MLP): TensorFlow `>= 2.12.0`.
- Install dependencies:

```
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Run the app

```
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser.

## Project Structure

- `app.py` — Flask application with routes, training, and persistence
- `templates/` — HTML templates
  - `workspace.html` — unified training/results/predict page
  - `register_model.html` — model creation/editing UI
  - `upload.html`, `map_columns.html` — dataset onboarding
  - `train.html`, `results.html`, `predict.html` — legacy pages (Workspace supersedes them)
- `static/js/main.js` — front-end behavior (training workflow, UI helpers)
- `static/css/style.css` — styles
- `data/` — uploaded datasets (CSV)
- `models/` — training artifacts and Keras architecture definitions
- `config.json` — persisted app state (datasets, models, trainings)
- `.gitignore` — excludes data and generated files from Git

## Usage Guide

### 1) Upload a Dataset
- Go to `Upload`.
- Select a CSV; upload.
- After upload, go to `Map Columns` to choose:
  - `features` — numeric columns used as input
  - `target` — classification target
  - Optional `label_map` — if you need to map string labels to ints

### 2) Register Models
- Go to `Register Model`.
- Create classic ML models (LogReg, SVM, RF, KNN, Naive Bayes) by filling their hyperparameters.
- Perceptron (`keras_mlp`):
  - Optimizer, Loss: dropdowns
  - Epochs, Batch size: integers
  - Hidden units: controlled via the architecture file (created automatically)
- You can edit existing models; saving redirects back to fresh create mode.
- The page lists all registered models with Edit/Delete actions.

### 3) Train & Compare in Workspace
- Go to `Workspace`.
- Select a dataset and one or more models; click `Train`.
- The training is async and shows progress (including epoch count for Keras).
- On completion, the page reloads and stays filtered on your selected dataset.
- Each card contains:
  - Confusion matrix with TN/FP/FN/TP counts and summary metrics
  - For Keras MLP: a `Parameters` section showing optimizer, loss, epochs, batch size, and hidden units
  - Optional loss chart when history is available (Keras)
- You can delete individual training runs; it removes associated files and updates `config.json`.

### 4) Predict
- In `Workspace`, choose a trained model using the dropdown.
- Enter feature values; click `Predict`.
- The predicted class appears inline next to the button.

## Model Details

### Classic ML
- Logistic Regression: `C`, `max_iter`.
- SVM: `C`, `kernel`, `gamma`.
- Random Forest: `n_estimators`, `max_depth`.
- KNN: `n_neighbors`, `weights`.
- Naive Bayes: Gaussian NB (no hyperparameters).

### Keras MLP (Perceptron)
- Architecture file: created when you register the perceptron.
- Example `architecture_*.json`:

```
{
  "layers": [
    { "Dense": { "units": 8, "activation": "relu" } },
    { "Dense": { "units": 1, "activation": "sigmoid" } }
  ]
}
```

- Hyperparameters in model:
  - `optimizer`: e.g., `Adam`, `SGD`, `RMSprop`, `Adagrad`, `Adamax`
  - `loss`: e.g., `binary_crossentropy`
  - `epochs`: integer
  - `batch_size`: integer
  - `learning_rate`: optional (used for optimizer construction)
- Training behavior:
  - Binary classification; targets are mapped to `{0,1}` internally for BCE.
  - Parameters are logged to the console and stored per training.

## Configuration (`config.json`)

- Top-level keys: `datasets`, `models`, `trainings`.
- Dataset entry (example):

```
{
  "id": "dataset_uuid",
  "name": "My Dataset",
  "path": "data/dataset_uuid.csv",
  "features": ["feat1", "feat2", "feat3"],
  "target": "target_col",
  "label_map": {"negative": -1, "positive": 1}
}
```

- Model entry (example for perceptron):

```
{
  "id": "261bd630-5311-412f-a9a0-5fb874dc0d56",
  "name": "Perceptron 1",
  "type": "keras_mlp",
  "created_at": "2025-11-11T13:46:02.724164",
  "hyperparams": {
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "epochs": 200,
    "batch_size": 32,
    "architecture_path": "models/architecture_261bd630-...json"
  }
}
```

- Training entry (example):

```
{
  "id": "training_uuid",
  "dataset_id": "dataset_uuid",
  "model_id": "model_uuid",
  "metrics": {
    "accuracy": 0.95,
    "precision_macro": 0.94,
    "recall_macro": 0.93,
    "f1_macro": 0.94,
    "confusion": {"tn": 10, "fp": 2, "fn": 1, "tp": 12}
  },
  "trained_path": "models/trained_sklearn_training_uuid.pkl",
  "preprocess_path": "models/preprocess_training_uuid.pkl",
  "history_path": "models/history_training_uuid.json",
  "params": {
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "epochs": 200,
    "batch_size": 32,
    "hidden_units": 8
  },
  "created_at": "2025-11-11T14:10:02.123Z"
}
```

## API Endpoints

- `GET /workspace` — main page; accepts `dataset_id`, `predict_model_id`, `predicted` in query
- `POST /workspace` — training or prediction depending on form fields
- `POST /api/train/start` — start async training

```
{
  "dataset_id": "...",
  "model_ids": ["...", "..."]
  // Optional overrides: epochs, batch_size, optimizer, loss, learning_rate
}
```

- `GET /api/train/status` — training status

```
{
  "status": "running|completed|error",
  "progress": 0.0..1.0,
  "messages": ["..."]
}
```

- `POST /api/train/stop` — request training stop
- `GET /api/history/<training_id>` — loss history (if available)
- `POST /training/<training_id>/delete` — delete a training record and its files
- `POST /model/<model_id>/delete` — delete a model and cascade-remove its trainings/files

## Persistence & Logging
- All state lives in `config.json`.
- Generated files live under `models/`.
- Console logs print training progress and resolved parameters (especially for Keras).

## Git Ignore & Data Hygiene
- `.gitignore` excludes:
  - `data/*`, `example datasets/*`
  - `models/preprocess_*.pkl`, `models/trained_*`, `models/history_*.json`
  - Python caches, virtual envs, build outputs, logs
- Architecture files stay tracked: `models/architecture_*.json`.
- Consider adding `.gitkeep` in `data/`, `models/` if you want empty folders tracked.

## Troubleshooting
- Perceptron epochs not honored: ensure your model’s `hyperparams.epochs` is set; the async trainer now prefers model hyperparams.
- Dataset filter resets after training: fixed — the Workspace now preserves `dataset_id` on reload after completion.
- KNN/LogReg training crashed with `opt_name` error: fixed — Keras-only params are scoped to Keras models.
- TensorFlow not installed: perceptron training is skipped with a message; install TensorFlow.

## Contributing
- Keep changes minimal and focused.
- Follow the existing style; prefer clear names over comments.
- Don’t commit datasets or training artifacts; `.gitignore` is set accordingly.

## License
- Not specified. Add one if you plan to share/distribute.