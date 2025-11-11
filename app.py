import os
import json
import uuid
from datetime import datetime
from urllib.parse import quote_plus

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import threading
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pickle


# Lazy imports for heavy libs inside functions to avoid startup failures

APP_SECRET = os.environ.get("APP_SECRET", "dev-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)


# Simple identity scaler used when datasets are pre-scaled in preprocessing wizard
class IdentityScaler:
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


def _normalize_name(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _detect_column_types(df):
    import pandas as pd
    types = {}
    for c in df.columns:
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            types[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[c] = "datetime"
        else:
            # Heuristic: low unique count => categorical; else text
            uniq = series.dropna().unique()
            if len(uniq) <= max(20, int(0.05 * len(series))):
                types[c] = "categorical"
            else:
                types[c] = "text"
    return types


def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = {"datasets": [], "models": [], "trainings": []}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return cfg
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"datasets": [], "models": [], "trainings": []}


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = APP_SECRET

# Training job state for async workflow
TRAIN_JOB = {
    "id": None,
    "status": "idle",  # idle | running | completed | error
    "progress": 0.0,
    "current_index": 0,
    "total": 0,
    "messages": [],
    "stop": False,
    "error": None,
    "epoch": 0,
    "epoch_total": 0
}


@app.context_processor
def inject_globals():
    # Provide current_year to all templates to avoid using undefined functions
    return {"current_year": datetime.utcnow().year}


@app.route("/")
def index():
    ensure_dirs()
    # Redirect to unified workspace; home page is no longer needed
    return redirect(url_for("workspace"))


# 2.1 Data Management: Upload
@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    ensure_dirs()
    cfg = load_config()
    if request.method == "GET":
        # Optional preview of a dataset
        view_id = request.args.get("view_id")
        view = None
        row_counts = {}
        if view_id:
            ds_meta = next((d for d in cfg.get("datasets", []) if d["id"] == view_id), None)
            if ds_meta:
                try:
                    import pandas as pd
                    ds_path = os.path.join(BASE_DIR, ds_meta["path"]) if not os.path.isabs(ds_meta["path"]) else ds_meta["path"]
                    df = pd.read_csv(ds_path)
                    row_counts[ds_meta["id"]] = int(len(df))
                    cols = list(df.columns)
                    rows = [ {c: (str(row[c]) if not pd.isna(row[c]) else "") for c in cols} for _, row in df.head(20).iterrows() ]
                    view = {"id": ds_meta["id"], "columns": cols, "rows": rows}
                except Exception:
                    view = None
        # Compute row counts for any datasets missing this info
        try:
            import pandas as pd
            for d in cfg.get("datasets", []):
                if d.get("row_count") is not None:
                    row_counts[d["id"]] = d.get("row_count")
                elif d["id"] not in row_counts:
                    p = os.path.join(BASE_DIR, d["path"]) if not os.path.isabs(d["path"]) else d["path"]
                    try:
                        df_tmp = pd.read_csv(p)
                        row_counts[d["id"]] = int(len(df_tmp))
                    except Exception:
                        row_counts[d["id"]] = None
        except Exception:
            pass
        return render_template("upload.html", datasets=cfg.get("datasets", []), view=view, row_counts=row_counts)

    # POST
    file = request.files.get("dataset_file")
    if not file or file.filename == "":
        flash("Please select a .csv, .xls, or .xlsx file to upload.")
        return redirect(url_for("upload_dataset"))

    filename = file.filename
    dataset_name = (request.form.get("dataset_name") or "").strip()
    ext = os.path.splitext(filename)[1].lower()

    # Read using pandas and save a standardized CSV copy
    try:
        import pandas as pd

        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in [".xls", ".xlsx"]:
            # Try reading Excel with openpyxl/xlrd depending on extension
            if ext == ".xlsx":
                df = pd.read_excel(file, engine="openpyxl")
            else:
                # .xls
                try:
                    df = pd.read_excel(file, engine="xlrd")
                except Exception:
                    # Fallback to pandas default (may fail if xlrd not available)
                    df = pd.read_excel(file)
        else:
            flash("Unsupported file type. Please upload .csv, .xls, or .xlsx.")
            return redirect(url_for("upload_dataset"))

        raw_id = str(uuid.uuid4())
        raw_path = os.path.join(DATA_DIR, f"raw_{raw_id}.csv")
        df.to_csv(raw_path, index=False)
        flash("File uploaded successfully. Proceed to preprocessing wizard.")
        # Pass through requested dataset name to wizard for intuitive naming
        qp = f"?dataset_name={quote_plus(dataset_name)}" if dataset_name else ""
        return redirect(url_for("preprocess_wizard", raw_id=raw_id) + qp)

    except Exception as e:
        flash(f"Error reading file: {e}")
        return redirect(url_for("upload_dataset"))


# Preprocessing Wizard: analyze and transform before final dataset save
@app.route("/preprocess/<raw_id>", methods=["GET", "POST"])
def preprocess_wizard(raw_id):
    ensure_dirs()
    raw_path = os.path.join(DATA_DIR, f"raw_{raw_id}.csv")
    if not os.path.exists(raw_path):
        flash("Uploaded file not found. Please re-upload.")
        return redirect(url_for("upload_dataset"))

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    import json as _json

    df = pd.read_csv(raw_path)
    columns = list(df.columns)
    column_types = _detect_column_types(df)

    # Detect chance column heuristically
    normalized = {c: _normalize_name(c) for c in columns}
    chance_col = None
    for c, n in normalized.items():
        if "chance" in n or "chanceofadmit" in n:
            chance_col = c
            break

    dataset_name = (request.args.get("dataset_name") or request.form.get("dataset_name") or "").strip()

    if request.method == "GET":
        chance_info = {}
        if chance_col:
            if column_types.get(chance_col) == "numeric":
                series = pd.to_numeric(df[chance_col], errors="coerce")
                chance_info = {
                    "is_numeric": True,
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                    "mean": float(series.mean(skipna=True)),
                    "median": float(series.median(skipna=True)),
                    "suggest_threshold": float(np.nanmedian(series)) if not np.isnan(series.median(skipna=True)) else 0.5,
                }
            else:
                uniq = list(pd.Series(df[chance_col]).dropna().unique())
                chance_info = {
                    "is_numeric": False,
                    "unique_values": uniq,
                }

        # Suggested features: all except detected chance column
        suggested_features = [c for c in columns if c != chance_col]

        # Sample rows for preview (raw)
        rows = [ {c: (str(row[c]) if not pd.isna(row[c]) else "") for c in columns} for _, row in df.head(20).iterrows() ]

        return render_template(
            "preprocess_wizard.html",
            raw_id=raw_id,
            columns=columns,
            column_types=column_types,
            chance_col=chance_col,
            chance_info=chance_info,
            dataset_name=dataset_name,
            suggested_features=suggested_features,
            preview=None,
            process_log=None,
        )

    # POST: Preview or Confirm transformations
    action = request.form.get("action")
    selected_features = request.form.getlist("features")
    target_col = request.form.get("target_col") or chance_col
    if not target_col:
        flash("Please specify a target column (e.g., chance).")
        return redirect(url_for("preprocess_wizard", raw_id=raw_id))

    is_numeric_flag = request.form.get("target_is_numeric")
    is_numeric = (is_numeric_flag == "1") if is_numeric_flag is not None else (column_types.get(target_col) == "numeric")

    # Build binary target based on threshold or mapping
    label_map = {}
    if is_numeric:
        try:
            threshold = float(request.form.get("chance_threshold") or 0.5)
        except Exception:
            threshold = 0.5
        series = pd.to_numeric(df[target_col], errors="coerce").fillna(threshold)
        y_std = np.where(series >= threshold, 1, -1)
        label_map = {"<" + str(threshold): -1, ">=" + str(threshold): 1}
        target_details = {"type": "numeric", "source": target_col, "threshold": threshold}
    else:
        # Collect mapping from form: map_value_X => -1/1
        uniq = list(pd.Series(df[target_col]).dropna().unique())
        chosen_map = {}
        for val in uniq:
            key = f"map_{val}"
            m = request.form.get(key)
            if m in ("-1", "1"):
                chosen_map[val] = int(m)
        # Fallback: map first unique to -1, rest to 1
        if not chosen_map:
            if len(uniq) >= 2:
                chosen_map = {uniq[0]: -1}
                for v in uniq[1:]:
                    chosen_map[v] = 1
            else:
                chosen_map = {uniq[0]: 1} if uniq else {}
        y_std = pd.Series(df[target_col]).map(lambda v: chosen_map.get(v, -1)).fillna(-1).astype(int).values
        label_map = {str(k): int(v) for k, v in chosen_map.items()}
        target_details = {"type": "categorical", "source": target_col, "mapping": label_map}

    # Assemble feature matrix with encodings
    encodings = {}
    X_trans = pd.DataFrame()
    working = df.copy()
    # Handle missing values upfront: numbers -> median, text -> empty string
    for c in working.columns:
        if column_types.get(c) == "numeric":
            try:
                med = pd.to_numeric(working[c], errors="coerce").median()
                working[c] = pd.to_numeric(working[c], errors="coerce").fillna(med)
            except Exception:
                working[c] = pd.to_numeric(working[c], errors="coerce").fillna(0.0)
        else:
            working[c] = working[c].astype(str).fillna("")

    # Use selected features, excluding target
    feature_cols = [c for c in selected_features if c != target_col]
    # If no features selected, default to all non-target columns
    if not feature_cols:
        feature_cols = [c for c in columns if c != target_col]

    for c in feature_cols:
        enc = request.form.get(f"encoding_{c}") or "none"
        encodings[c] = enc
        if enc == "onehot" and column_types.get(c) in ("categorical", "text"):
            dummies = pd.get_dummies(working[c], prefix=f"ohe_{c}")
            X_trans = pd.concat([X_trans, dummies], axis=1)
        elif enc == "label" and column_types.get(c) in ("categorical", "text"):
            values = list(pd.Series(working[c]).dropna().unique())
            map_lbl = {v: i for i, v in enumerate(values)}
            X_trans[f"le_{c}"] = pd.Series(working[c]).map(lambda v: map_lbl.get(v, 0)).astype(float)
            encodings[c] = {"method": "label", "map": map_lbl}
        elif enc == "tfidf" and column_types.get(c) == "text":
            tfv = TfidfVectorizer(max_features=50)
            mat = tfv.fit_transform(working[c].astype(str).fillna(""))
            tf_cols = [f"tfidf_{c}_{i}" for i in range(mat.shape[1])]
            X_tfidf = pd.DataFrame(mat.toarray(), columns=tf_cols)
            X_trans = pd.concat([X_trans, X_tfidf], axis=1)
            encodings[c] = {"method": "tfidf", "vocab_size": int(mat.shape[1])}
        else:
            # numeric or no encoding for text/categorical -> attempt numeric cast
            if column_types.get(c) == "numeric":
                X_trans[c] = pd.to_numeric(working[c], errors="coerce").fillna(0.0)
            else:
                # treat as label with fallback
                values = list(pd.Series(working[c]).dropna().unique())
                map_lbl = {v: i for i, v in enumerate(values)}
                X_trans[f"le_{c}"] = pd.Series(working[c]).map(lambda v: map_lbl.get(v, 0)).astype(float)
                encodings[c] = {"method": "label", "map": map_lbl}

    # Scaling (global method)
    scaling_method = (request.form.get("scaling_method") or "none").lower()
    scaler_applied = None
    if scaling_method in ("standard", "minmax", "robust"):
        if scaling_method == "standard":
            scaler_applied = StandardScaler()
        elif scaling_method == "minmax":
            scaler_applied = MinMaxScaler()
        else:
            scaler_applied = RobustScaler()
        X_trans_np = scaler_applied.fit_transform(X_trans.values.astype(float))
        X_trans = pd.DataFrame(X_trans_np, columns=list(X_trans.columns))

    # Build final transformed dataframe
    target_name = "target_binary"
    std_df = X_trans.copy()
    std_df[target_name] = y_std

    # Validation: ensure all features numeric and no NaNs
    if std_df.drop(columns=[target_name]).isna().any().any():
        flash("Invalid or missing values detected after preprocessing. Please adjust transformations.")
        # Show preview so user can iterate
        preview_rows = [ {c: (str(row[c]) if not pd.isna(row[c]) else "") for c in std_df.columns} for _, row in std_df.head(20).iterrows() ]
        process_log = {
            "column_types": column_types,
            "target": target_details,
            "encodings": encodings,
            "scaling": scaling_method,
            "notes": ["Detected NaN values post-transform"]
        }
        return render_template(
            "preprocess_wizard.html",
            raw_id=raw_id,
            columns=columns,
            column_types=column_types,
            chance_col=target_col,
            chance_info=None,
            dataset_name=dataset_name,
            suggested_features=feature_cols,
            preview={"columns": list(std_df.columns), "rows": preview_rows},
            process_log=process_log,
        )

    # Prepare processing log
    missing_counts = {c: int(X_trans[c].isna().sum()) for c in X_trans.columns}
    process_log = {
        "column_types": column_types,
        "target": target_details,
        "encodings": encodings,
        "scaling": scaling_method,
        "missing_counts": missing_counts,
        "feature_columns_out": list(X_trans.columns),
    }

    if action == "preview":
        preview_rows = [ {c: (str(row[c]) if not pd.isna(row[c]) else "") for c in std_df.columns} for _, row in std_df.head(20).iterrows() ]
        # Persist temporary log for review
        tmp_log_path = os.path.join(DATA_DIR, f"process_log_raw_{raw_id}.json")
        try:
            with open(tmp_log_path, "w", encoding="utf-8") as lf:
                _json.dump(process_log, lf, indent=2)
        except Exception:
            pass
        return render_template(
            "preprocess_wizard.html",
            raw_id=raw_id,
            columns=columns,
            column_types=column_types,
            chance_col=target_col,
            chance_info=None,
            dataset_name=dataset_name,
            suggested_features=feature_cols,
            preview={"columns": list(std_df.columns), "rows": preview_rows},
            process_log=process_log,
        )

    # Confirm: save dataset and metadata
    dataset_id = str(uuid.uuid4())
    dataset_path = os.path.join(DATA_DIR, f"dataset_{dataset_id}.csv")
    std_df.to_csv(dataset_path, index=False)
    row_count = int(len(std_df))

    # Preserve original uploaded file alongside transformed dataset
    import shutil
    original_copy_path = os.path.join(DATA_DIR, f"original_{dataset_id}.csv")
    try:
        shutil.copyfile(raw_path, original_copy_path)
    except Exception:
        original_copy_path = None

    # Save log to persistent location tied to dataset_id
    log_path = os.path.join(DATA_DIR, f"process_log_{dataset_id}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            _json.dump(process_log, lf, indent=2)
    except Exception:
        log_path = None

    cfg = load_config()
    cfg.setdefault("datasets", []).append({
        "id": dataset_id,
        "name": dataset_name if dataset_name else f"Dataset {dataset_id[:8]}",
        "path": os.path.relpath(dataset_path, BASE_DIR).replace("\\", "/"),
        "original_path": (os.path.relpath(original_copy_path, BASE_DIR).replace("\\", "/") if original_copy_path else None),
        "features": list(X_trans.columns),
        "target": target_name,
        "label_map": label_map,
        "row_count": row_count,
        "preprocessing": {
            "scaling": scaling_method if scaling_method != "none" else None,
            "encodings": encodings,
            "target": target_details,
            "column_types": column_types,
            "process_log_path": (os.path.relpath(log_path, BASE_DIR).replace("\\", "/") if log_path else None)
        },
        "created_at": datetime.utcnow().isoformat()
    })
    save_config(cfg)

    # Remove temporary raw upload now that original has been preserved
    try:
        os.remove(raw_path)
    except Exception:
        pass

    flash("Dataset preprocessed, standardized, and saved.")
    return redirect(url_for("workspace"))


@app.route("/map_columns/<raw_id>", methods=["GET", "POST"])
def map_columns(raw_id):
    ensure_dirs()
    raw_path = os.path.join(DATA_DIR, f"raw_{raw_id}.csv")
    if not os.path.exists(raw_path):
        flash("Uploaded file not found. Please re-upload.")
        return redirect(url_for("upload_dataset"))

    import pandas as pd
    df = pd.read_csv(raw_path)
    columns = list(df.columns)

    if request.method == "GET":
        # Suggest first 3 columns as features, last as target
        suggested_features = columns[:3] if len(columns) >= 3 else columns[:-1]
        suggested_target = columns[-1] if len(columns) > 0 else None
        sample_target_values = []
        if suggested_target:
            sample_target_values = list(pd.Series(df[suggested_target]).dropna().unique()[:5])
        return render_template(
            "map_columns.html",
            raw_id=raw_id,
            columns=columns,
            suggested_features=suggested_features,
            suggested_target=suggested_target,
            sample_target_values=sample_target_values,
        )

    # POST: receive mapping
    features = request.form.getlist("features")
    target = request.form.get("target")
    dataset_name = (request.form.get("dataset_name") or request.args.get("dataset_name") or "").strip()
    if not features or not target:
        flash("Please select feature columns and target column.")
        return redirect(url_for("map_columns", raw_id=raw_id))

    # Standardize target to -1/+1
    y_series = df[target]
    uniques = sorted([u for u in pd.Series(y_series).dropna().unique()])
    label_map = {}

    if set(uniques) == {-1, 1}:
        label_map = {-1: -1, 1: 1}
        y_std = y_series
    elif len(uniques) == 2:
        # Map deterministically: min -> -1, max -> +1
        low, high = uniques[0], uniques[1]
        label_map = {low: -1, high: 1}
        y_std = y_series.map(lambda v: label_map.get(v, -1))
    else:
        flash("Target column must contain exactly two classes. Please choose another target column.")
        return redirect(url_for("map_columns", raw_id=raw_id))

    # Build standardized dataset with selected features and standardized target
    X_df = df[features].copy()
    std_df = X_df.copy()
    std_df[target] = y_std

    dataset_id = str(uuid.uuid4())
    dataset_path = os.path.join(DATA_DIR, f"dataset_{dataset_id}.csv")
    std_df.to_csv(dataset_path, index=False)
    row_count = int(len(std_df))

    # Update config
    cfg = load_config()
    cfg["datasets"].append({
        "id": dataset_id,
        "name": dataset_name if dataset_name else f"Dataset {dataset_id[:8]}",
        "path": os.path.relpath(dataset_path, BASE_DIR).replace("\\", "/"),
        "features": features,
        "target": target,
        "label_map": label_map,
        "row_count": row_count,
        "created_at": datetime.utcnow().isoformat()
    })
    save_config(cfg)

    # Optional: remove raw
    try:
        os.remove(raw_path)
    except Exception:
        pass

    flash("Dataset standardized and saved.")
    return redirect(url_for("workspace"))


# Remap columns for an existing standardized dataset
@app.route("/map_columns_existing/<dataset_id>", methods=["GET", "POST"])
def map_columns_existing(dataset_id):
    ensure_dirs()
    cfg = load_config()
    ds_meta = next((d for d in cfg.get("datasets", []) if d["id"] == dataset_id), None)
    if not ds_meta:
        flash("Dataset not found.")
        return redirect(url_for("upload_dataset"))

    import pandas as pd
    ds_path = os.path.join(BASE_DIR, ds_meta["path"]) if not os.path.isabs(ds_meta["path"]) else ds_meta["path"]
    df = pd.read_csv(ds_path)
    columns = list(df.columns)

    if request.method == "GET":
        suggested_features = ds_meta.get("features", columns[:-1])
        suggested_target = ds_meta.get("target", columns[-1] if columns else None)
        sample_target_values = []
        if suggested_target:
            sample_target_values = list(pd.Series(df[suggested_target]).dropna().unique()[:5])
        return render_template(
            "map_columns.html",
            columns=columns,
            suggested_features=suggested_features,
            suggested_target=suggested_target,
            sample_target_values=sample_target_values,
        )

    # POST: update mapping
    features = request.form.getlist("features")
    target = request.form.get("target")
    if not features or not target:
        flash("Please select feature columns and target column.")
        return redirect(url_for("map_columns_existing", dataset_id=dataset_id))

    # Standardize target to -1/+1 (dataset is already standardized, but allow switching target)
    y_series = df[target]
    uniques = sorted([u for u in pd.Series(y_series).dropna().unique()])
    label_map = {}
    if set(uniques) == {-1, 1}:
        label_map = {-1: -1, 1: 1}
        y_std = y_series
    elif len(uniques) == 2:
        low, high = uniques[0], uniques[1]
        label_map = {low: -1, high: 1}
        y_std = y_series.map(lambda v: label_map.get(v, -1))
    else:
        flash("Target column must contain exactly two classes. Please choose another target column.")
        return redirect(url_for("map_columns_existing", dataset_id=dataset_id))

    # Overwrite standardized dataset with new selection
    X_df = df[features].copy()
    std_df = X_df.copy()
    std_df[target] = y_std
    std_df.to_csv(ds_path, index=False)

    # Update config metadata
    ds_meta["features"] = features
    ds_meta["target"] = target
    ds_meta["label_map"] = label_map
    save_config(cfg)
    flash("Dataset mapping updated.")
    return redirect(url_for("upload_dataset", view_id=dataset_id))


# Dataset management: rename
@app.route("/dataset/<dataset_id>/rename", methods=["POST"])
def rename_dataset(dataset_id):
    ensure_dirs()
    cfg = load_config()
    new_name = (request.form.get("new_name") or "").strip()
    if not new_name:
        flash("New name is required.")
        return redirect(url_for("upload_dataset"))
    for d in cfg.get("datasets", []):
        if d["id"] == dataset_id:
            d["name"] = new_name
            break
    save_config(cfg)
    flash("Dataset renamed.")
    return redirect(url_for("upload_dataset", view_id=dataset_id))


@app.route("/dataset/<dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id):
    ensure_dirs()
    cfg = load_config()
    datasets = cfg.get("datasets", [])
    target_ds = next((d for d in datasets if d["id"] == dataset_id), None)
    if not target_ds:
        flash("Dataset not found.")
        return redirect(url_for("upload_dataset"))
    
    # Remove dataset and associated trainings
    cfg["datasets"] = [d for d in datasets if d["id"] != dataset_id]
    cfg["trainings"] = [t for t in cfg.get("trainings", []) if t["dataset_id"] != dataset_id]
    save_config(cfg)
    # Remove file
    try:
        ds_path = os.path.join(BASE_DIR, target_ds["path"]) if not os.path.isabs(target_ds["path"]) else target_ds["path"]
        if os.path.exists(ds_path):
            os.remove(ds_path)
    except Exception:
        pass

    flash("Dataset deleted.")
    return redirect(url_for("upload_dataset"))


# Unified workspace: merge train, results, predict
@app.route("/workspace", methods=["GET", "POST"])
def workspace():
    ensure_dirs()
    cfg = load_config()
    datasets = cfg.get("datasets", [])
    models = cfg.get("models", [])
    trainings = cfg.get("trainings", [])

    if request.method == "GET":
        selected_dataset_id = request.args.get("dataset_id") or (datasets[0]["id"] if datasets else None)
        selected_model_id = request.args.get("predict_model_id")
        predicted_value = request.args.get("predicted")
        try:
            predicted_value = int(predicted_value) if predicted_value is not None else None
        except Exception:
            predicted_value = None
        # Build latest trainings per model for selected dataset
        if selected_dataset_id:
            ds_trainings = [t for t in trainings if t["dataset_id"] == selected_dataset_id]
        else:
            ds_trainings = trainings

        latest_by_model = {}
        for t in ds_trainings:
            latest_by_model[t["model_id"]] = t
        latest_list = [
            {"model": next((m for m in models if m["id"] == mid), None), "training": tr}
            for mid, tr in latest_by_model.items()
            if any(m["id"] == mid for m in models)
        ]

        # Auto-select first available trained model for prediction if none chosen
        if not selected_model_id and latest_by_model:
            try:
                selected_model_id = next(iter(latest_by_model.keys()))
            except StopIteration:
                selected_model_id = None

        # Prediction feature columns for selected model
        feature_columns = []
        if selected_model_id:
            latest_train_by_model = {}
            for t in trainings:
                latest_train_by_model[t["model_id"]] = t
            selected_training = latest_train_by_model.get(selected_model_id)
            if selected_training:
                import pickle
                pp_path = os.path.join(BASE_DIR, selected_training["preprocess_path"]) if not os.path.isabs(selected_training["preprocess_path"]) else selected_training["preprocess_path"]
                try:
                    with open(pp_path, "rb") as pf:
                        pp = pickle.load(pf)
                        feature_columns = pp.get("feature_columns", [])
                except Exception:
                    feature_columns = []

        return render_template(
            "workspace.html",
            datasets=datasets,
            models=models,
            selected_dataset_id=selected_dataset_id,
            results=latest_list,
            feature_columns=feature_columns,
            selected_model_id=selected_model_id,
            predicted_value=predicted_value,
        )

    # POST: either training or prediction
    action = request.form.get("action")
    if action == "predict":
        selected_dataset_id = request.form.get("dataset_id")
        selected_model_id = request.form.get("predict_model_id")
        # Use the same logic as /predict
        # If no model explicitly chosen, auto-select first available trained model for this dataset
        if not selected_model_id:
            ds_trainings = [t for t in trainings if t["dataset_id"] == selected_dataset_id] if selected_dataset_id else trainings
            latest_by_model = {}
            for t in ds_trainings:
                latest_by_model[t["model_id"]] = t
            try:
                selected_model_id = next(iter(latest_by_model.keys()))
            except StopIteration:
                selected_model_id = None

        latest_train_by_model = {}
        for t in trainings:
            latest_train_by_model[t["model_id"]] = t
        selected_training = latest_train_by_model.get(selected_model_id)
        if not selected_training:
            flash("Please select a trained model.")
            return redirect(url_for("workspace", dataset_id=selected_dataset_id))
        import numpy as np
        import pickle
        pp_path = os.path.join(BASE_DIR, selected_training["preprocess_path"]) if not os.path.isabs(selected_training["preprocess_path"]) else selected_training["preprocess_path"]
        with open(pp_path, "rb") as pf:
            pp = pickle.load(pf)
        feature_columns = pp.get("feature_columns", [])
        try:
            X_input = [float(request.form.get(f"feature_{col}")) for col in feature_columns]
        except Exception:
            flash("Please enter numeric values for all features.")
            return redirect(url_for("workspace", dataset_id=selected_dataset_id, predict_model_id=selected_model_id))
        scaler = pp["scaler"]
        X_scaled = scaler.transform(np.array([X_input]))
        model_meta = next((m for m in models if m["id"] == selected_model_id), None)
        pred_class = None
        if model_meta["type"] == "keras_mlp":
            try:
                import tensorflow as tf
                from tensorflow import keras
            except Exception as e:
                flash(f"TensorFlow/Keras not available: {e}.")
                return redirect(url_for("workspace", dataset_id=selected_dataset_id))
            arch_path_rel = model_meta["hyperparams"]["architecture_path"]
            arch_path = os.path.join(BASE_DIR, arch_path_rel) if not os.path.isabs(arch_path_rel) else arch_path_rel
            with open(arch_path, "r", encoding="utf-8") as f:
                architecture = json.load(f)
            model = keras.Sequential()
            input_dim = X_scaled.shape[1]
            for layer_def in architecture["layers"]:
                if "Dense" in layer_def:
                    params = layer_def["Dense"]
                    units = int(params.get("units", 8))
                    activation = params.get("activation", "relu")
                    if len(model.layers) == 0:
                        model.add(keras.layers.Dense(units=units, activation=activation, input_shape=(input_dim,)))
                    else:
                        model.add(keras.layers.Dense(units=units, activation=activation))
            model.compile(optimizer=model_meta["hyperparams"].get("optimizer", "Adam"), loss=model_meta["hyperparams"].get("loss", "binary_crossentropy"))
            weights_path = os.path.join(BASE_DIR, selected_training["trained_path"]) if not os.path.isabs(selected_training["trained_path"]) else selected_training["trained_path"]
            if weights_path.endswith(".h5") and not weights_path.endswith(".weights.h5"):
                alt_path = weights_path[:-3] + ".weights.h5"
                try:
                    if os.path.exists(weights_path) and not os.path.exists(alt_path):
                        os.rename(weights_path, alt_path)
                    cfg = load_config()
                    for t in cfg.get("trainings", []):
                        if t.get("id") == selected_training.get("id"):
                            t["trained_path"] = os.path.relpath(alt_path, BASE_DIR)
                            save_config(cfg)
                            selected_training["trained_path"] = t["trained_path"]
                            break
                    weights_path = alt_path
                except Exception:
                    weights_path = alt_path
            model.load_weights(weights_path)
            prob = float(model.predict(X_scaled, verbose=0).ravel()[0])
            pred_class = 1 if prob >= 0.5 else -1
        else:
            mdl_path = os.path.join(BASE_DIR, selected_training["trained_path"]) if not os.path.isabs(selected_training["trained_path"]) else selected_training["trained_path"]
            with open(mdl_path, "rb") as mf:
                clf = pickle.load(mf)
            pred_class = int(clf.predict(X_scaled)[0])
        return redirect(url_for("workspace", dataset_id=selected_dataset_id, predict_model_id=selected_model_id, predicted=pred_class))

    # Training POST
    dataset_id = request.form.get("dataset_id")
    selected_model_ids = request.form.getlist("model_ids")
    if not dataset_id or not selected_model_ids:
        flash("Please select a dataset and at least one model.")
        return redirect(url_for("workspace"))

    ds_meta = next((d for d in datasets if d["id"] == dataset_id), None)
    if not ds_meta:
        flash("Dataset not found.")
        return redirect(url_for("workspace"))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import numpy as np
    import pickle

    dataset_path = os.path.join(BASE_DIR, ds_meta["path"]) if not os.path.isabs(ds_meta["path"]) else ds_meta["path"]
    df = pd.read_csv(dataset_path)
    X = df[ds_meta["features"]].values.astype(float)
    y = df[ds_meta["target"]].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Honor pre-scaling from preprocessing wizard
    preproc = ds_meta.get("preprocessing", {}) if ds_meta else {}
    pre_scaled = bool(preproc.get("scaling"))
    if pre_scaled:
        scaler = IdentityScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    for mid in selected_model_ids:
        model_meta = next((m for m in models if m["id"] == mid), None)
        if not model_meta:
            continue
        training_id = str(uuid.uuid4())
        preprocess_path = os.path.join(MODELS_DIR, f"preprocess_{training_id}.pkl")
        with open(preprocess_path, "wb") as pf:
            pickle.dump({
                "scaler": scaler,
                "feature_columns": ds_meta["features"],
                "target": ds_meta["target"],
                "label_map": ds_meta.get("label_map", {})
            }, pf)
        metrics = {}
        history_path = None
        trained_path = None
        if model_meta["type"] == "keras_mlp":
            try:
                import tensorflow as tf
                from tensorflow import keras
            except Exception as e:
                flash(f"TensorFlow/Keras not available: {e}. Skipping Keras model '{model_meta['name']}'.")
                continue
            # Build model
            arch_path_rel = model_meta["hyperparams"]["architecture_path"]
            arch_path = os.path.join(BASE_DIR, arch_path_rel) if not os.path.isabs(arch_path_rel) else arch_path_rel
            with open(arch_path, "r", encoding="utf-8") as f:
                architecture = json.load(f)
            model = keras.Sequential()
            input_dim = X_train_scaled.shape[1]
            for layer_def in architecture["layers"]:
                if "Dense" in layer_def:
                    params = layer_def["Dense"]
                    units = int(params.get("units", 8))
                    activation = params.get("activation", "relu")
                    if len(model.layers) == 0:
                        model.add(keras.layers.Dense(units=units, activation=activation, input_shape=(input_dim,)))
                    else:
                        model.add(keras.layers.Dense(units=units, activation=activation))
            model.add(keras.layers.Dense(units=1, activation="sigmoid"))
            optimizer = model_meta["hyperparams"].get("optimizer", "Adam")
            loss_fn = model_meta["hyperparams"].get("loss", "binary_crossentropy")
            epochs = int(model_meta["hyperparams"].get("epochs", 100))
            batch_size = int(model_meta["hyperparams"].get("batch_size", 32))
            print(f"[TRAIN] (workspace) Keras MLP params: optimizer={optimizer}, loss={loss_fn}, epochs={epochs}, batch_size={batch_size}")
            model.compile(optimizer=optimizer, loss=loss_fn)
            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
            history_path = os.path.join(MODELS_DIR, f"history_{training_id}.json")
            with open(history_path, "w", encoding="utf-8") as hf:
                json.dump({"loss": history.history.get("loss", []), "val_loss": history.history.get("val_loss", [])}, hf, indent=2)
            y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
            y_pred = np.where(y_pred_prob >= 0.5, 1, -1)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, pos_label=1))
            rec = float(recall_score(y_test, y_pred, pos_label=1))
            f1 = float(f1_score(y_test, y_pred, pos_label=1))
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}}
            # Save Keras weights with required extension
            trained_path = os.path.join(MODELS_DIR, f"trained_keras_{training_id}.weights.h5")
            model.save_weights(trained_path)
        else:
            clf = None
            if model_meta["type"] == "logistic_regression":
                clf = LogisticRegression(C=float(model_meta["hyperparams"].get("C", 1.0)), solver=model_meta["hyperparams"].get("solver", "liblinear"), max_iter=int(model_meta["hyperparams"].get("max_iter", 100)))
            elif model_meta["type"] == "svm":
                clf = SVC(C=float(model_meta["hyperparams"].get("C", 1.0)), kernel=model_meta["hyperparams"].get("kernel", "rbf"), probability=False)
            elif model_meta["type"] == "random_forest":
                md = model_meta["hyperparams"].get("max_depth", None)
                md = int(md) if (md and str(md).isdigit()) else None
                clf = RandomForestClassifier(n_estimators=int(model_meta["hyperparams"].get("n_estimators", 100)), max_depth=md)
            elif model_meta["type"] == "knn":
                clf = KNeighborsClassifier(n_neighbors=int(model_meta["hyperparams"].get("n_neighbors", 5)))
            elif model_meta["type"] == "naive_bayes":
                clf = GaussianNB()
            else:
                continue
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, pos_label=1))
            rec = float(recall_score(y_test, y_pred, pos_label=1))
            f1 = float(f1_score(y_test, y_pred, pos_label=1))
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}}
            trained_path = os.path.join(MODELS_DIR, f"trained_sklearn_{training_id}.pkl")
            with open(trained_path, "wb") as mf:
                pickle.dump(clf, mf)

        # Extract hidden_units from first Dense layer for logging
        try:
            first_dense = next((ld["Dense"] for ld in architecture["layers"] if "Dense" in ld), {})
            hidden_units_used = int(first_dense.get("units", 8))
        except Exception:
            hidden_units_used = None

        params_record = {}
        if model_meta["type"] == "keras_mlp":
            params_record = {"optimizer": optimizer, "loss": loss_fn, "epochs": epochs, "batch_size": batch_size, "hidden_units": hidden_units_used}

        cfg["trainings"].append({
            "id": training_id,
            "dataset_id": dataset_id,
            "model_id": mid,
            "metrics": metrics,
            "trained_path": os.path.relpath(trained_path, BASE_DIR).replace("\\", "/") if trained_path else None,
            "preprocess_path": os.path.relpath(preprocess_path, BASE_DIR).replace("\\", "/"),
            "history_path": os.path.relpath(history_path, BASE_DIR).replace("\\", "/") if history_path else None,
            "params": params_record,
            "created_at": datetime.utcnow().isoformat()
        })

    save_config(cfg)
    flash("Training complete.")
    return redirect(url_for("workspace", dataset_id=dataset_id))


# 2.2 Model Management: Registration
@app.route("/register_model", methods=["GET", "POST"])
def register_model():
    ensure_dirs()
    cfg = load_config()
    if request.method == "GET":
        models = cfg.get("models", [])
        edit_id = request.args.get("model_id")
        editing_model = next((m for m in models if m.get("id") == edit_id), None) if edit_id else None
        return render_template("register_model.html", models=models, editing_model=editing_model)

    # POST
    model_name = request.form.get("model_name")
    model_type = request.form.get("model_type")
    if not model_name or not model_type:
        flash("Model Name and Model Type are required.")
        return redirect(url_for("register_model"))

    edit_id = request.form.get("model_id")
    if edit_id:
        entry = next((m for m in cfg.get("models", []) if m.get("id") == edit_id), None)
        if not entry:
            flash("Model not found.")
            return redirect(url_for("register_model"))
        entry["name"] = model_name
        entry["type"] = model_type
    else:
        model_id = str(uuid.uuid4())
        entry = {
            "id": model_id,
            "name": model_name,
            "type": model_type,
            "created_at": datetime.utcnow().isoformat()
        }

    # Collect hyperparams with defaults
    if model_type == "keras_mlp":
        # Architecture: shallow MLP default
        hidden_units = int(request.form.get("hidden_units", "8"))
        optimizer = request.form.get("optimizer", "Adam")
        loss = request.form.get("loss", "binary_crossentropy")
        epochs = int(request.form.get("epochs", "100"))
        batch_size = int(request.form.get("batch_size", "32"))

        architecture = {
            "type": "Sequential",
            "layers": [
                {"Dense": {"units": hidden_units, "activation": "relu"}},
                {"Dense": {"units": 1, "activation": "sigmoid"}}
            ]
        }

        # Write/update architecture JSON
        if edit_id:
            arch_rel = (entry.get("hyperparams") or {}).get("architecture_path")
            arch_path = os.path.join(BASE_DIR, arch_rel) if (arch_rel and not os.path.isabs(arch_rel)) else (arch_rel or os.path.join(MODELS_DIR, f"architecture_{edit_id}.json"))
            try:
                with open(arch_path, "w", encoding="utf-8") as f:
                    json.dump(architecture, f, indent=2)
            except Exception:
                pass
            arch_rel_new = os.path.relpath(arch_path, BASE_DIR).replace("\\", "/")
        else:
            arch_path = os.path.join(MODELS_DIR, f"architecture_{entry['id']}.json")
            with open(arch_path, "w", encoding="utf-8") as f:
                json.dump(architecture, f, indent=2)
            arch_rel_new = os.path.relpath(arch_path, BASE_DIR).replace("\\", "/")

        entry["hyperparams"] = {
            "optimizer": optimizer,
            "loss": loss,
            "epochs": epochs,
            "batch_size": batch_size,
            "architecture_path": arch_rel_new
        }

    elif model_type == "logistic_regression":
        entry["hyperparams"] = {
            "C": float(request.form.get("C", "1.0")),
            "solver": request.form.get("solver", "liblinear"),
            "max_iter": int(request.form.get("max_iter", "100"))
        }
    elif model_type == "svm":
        entry["hyperparams"] = {
            "C": float(request.form.get("C", "1.0")),
            "kernel": request.form.get("kernel", "rbf"),
        }
    elif model_type == "random_forest":
        entry["hyperparams"] = {
            "n_estimators": int(request.form.get("n_estimators", "100")),
            "max_depth": request.form.get("max_depth", "") or None,
        }
    elif model_type == "knn":
        entry["hyperparams"] = {
            "n_neighbors": int(request.form.get("n_neighbors", "5"))
        }
    elif model_type == "naive_bayes":
        entry["hyperparams"] = {}
    else:
        flash("Unsupported model type.")
        return redirect(url_for("register_model"))

    models = cfg.setdefault("models", [])
    if edit_id:
        for i, m in enumerate(models):
            if m.get("id") == entry.get("id"):
                models[i] = entry
                break
        flash("Model updated.")
    else:
        models.append(entry)
        flash("Model registered successfully.")
    save_config(cfg)
    # After update or registration, return to fresh form (no editing state)
    return redirect(url_for("register_model"))


@app.route("/model/<model_id>/delete", methods=["POST"])
def model_delete(model_id):
    """Delete a registered model and cascade-remove its trainings and files."""
    ensure_dirs()
    cfg = load_config()
    models = cfg.get("models", [])
    m = next((mm for mm in models if mm.get("id") == model_id), None)
    if not m:
        flash("Model not found.")
        return redirect(url_for("register_model"))
    # Remove architecture file if Keras model
    if m.get("type") == "keras_mlp":
        arch_rel = (m.get("hyperparams") or {}).get("architecture_path")
        if arch_rel:
            p = os.path.join(BASE_DIR, arch_rel) if not os.path.isabs(arch_rel) else arch_rel
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    # Remove trainings for this model and associated files
    kept_trainings = []
    for t in cfg.get("trainings", []):
        if t.get("model_id") == model_id:
            for key in ("trained_path", "preprocess_path", "history_path"):
                p = t.get(key)
                if not p:
                    continue
                p_abs = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
                try:
                    if os.path.exists(p_abs):
                        os.remove(p_abs)
                except Exception:
                    pass
        else:
            kept_trainings.append(t)
    cfg["trainings"] = kept_trainings
    cfg["models"] = [mm for mm in models if mm.get("id") != model_id]
    save_config(cfg)
    flash("Model deleted.")
    return redirect(url_for("register_model"))


# Training & Evaluation
@app.route("/train", methods=["GET", "POST"])
def train():
    ensure_dirs()
    cfg = load_config()
    datasets = cfg.get("datasets", [])
    models = cfg.get("models", [])

    if request.method == "GET":
        return render_template("train.html", datasets=datasets, models=models)

    dataset_id = request.form.get("dataset_id")
    selected_model_ids = request.form.getlist("model_ids")
    if not dataset_id or not selected_model_ids:
        flash("Please select a dataset and at least one model.")
        return redirect(url_for("train"))

    # Load dataset metadata
    ds_meta = next((d for d in datasets if d["id"] == dataset_id), None)
    if not ds_meta:
        flash("Dataset not found.")
        return redirect(url_for("train"))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import numpy as np
    import pickle

    dataset_path = os.path.join(BASE_DIR, ds_meta["path"]) if not os.path.isabs(ds_meta["path"]) else ds_meta["path"]
    df = pd.read_csv(dataset_path)
    X = df[ds_meta["features"]].values.astype(float)
    y = df[ds_meta["target"]].values.astype(int)

    # Preprocessing: scaling and split (honor pre-scaled datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preproc = ds_meta.get("preprocessing", {}) if ds_meta else {}
    pre_scaled = bool(preproc.get("scaling"))
    if pre_scaled:
        scaler = IdentityScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    results = []

    for mid in selected_model_ids:
        model_meta = next((m for m in models if m["id"] == mid), None)
        if not model_meta:
            continue

        training_id = str(uuid.uuid4())
        preprocess_path = os.path.join(MODELS_DIR, f"preprocess_{training_id}.pkl")
        with open(preprocess_path, "wb") as pf:
            pickle.dump({
                "scaler": scaler,
                "feature_columns": ds_meta["features"],
                "target": ds_meta["target"],
                "label_map": ds_meta.get("label_map", {}),
                "pre_scaled": pre_scaled,
                "scaling_method": preproc.get("scaling")
            }, pf)

        metrics = {}
        history_path = None
        trained_path = None

        if model_meta["type"] == "keras_mlp":
            # Lazy import tensorflow/keras with error handling
            try:
                import tensorflow as tf
                from tensorflow import keras
            except Exception as e:
                flash(f"TensorFlow/Keras not available: {e}. Skipping Keras model '{model_meta['name']}'.")
                continue

            # Build model from saved architecture
            arch_path_rel = model_meta["hyperparams"]["architecture_path"]
            arch_path = os.path.join(BASE_DIR, arch_path_rel) if not os.path.isabs(arch_path_rel) else arch_path_rel
            with open(arch_path, "r", encoding="utf-8") as f:
                architecture = json.load(f)

            model = keras.Sequential()
            input_dim = X_train_scaled.shape[1]
            for layer_def in architecture["layers"]:
                if "Dense" in layer_def:
                    params = layer_def["Dense"]
                    units = int(params.get("units", 8))
                    activation = params.get("activation", "relu")
                    if len(model.layers) == 0:
                        model.add(keras.layers.Dense(units=units, activation=activation, input_shape=(input_dim,)))
                    else:
                        model.add(keras.layers.Dense(units=units, activation=activation))

            optimizer = model_meta["hyperparams"].get("optimizer", "Adam")
            loss = model_meta["hyperparams"].get("loss", "binary_crossentropy")
            epochs = int(model_meta["hyperparams"].get("epochs", 100))
            batch_size = int(model_meta["hyperparams"].get("batch_size", 32))
            print(f"[TRAIN] (/train) Keras MLP params: optimizer={optimizer}, loss={loss}, epochs={epochs}, batch_size={batch_size}")
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

            # Keras expects targets in {0,1} for BCE
            y_train_01 = ((y_train + 1) // 2).astype(int)
            y_test_01 = ((y_test + 1) // 2).astype(int)

            history = model.fit(
                X_train_scaled, y_train_01,
                validation_data=(X_test_scaled, y_test_01),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # Save training history
            history_path = os.path.join(MODELS_DIR, f"history_{training_id}.json")
            with open(history_path, "w", encoding="utf-8") as hf:
                json.dump({
                    "loss": history.history.get("loss", []),
                    "val_loss": history.history.get("val_loss", [])
                }, hf, indent=2)

            # Predictions and metrics
            y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
            y_pred = np.where(y_pred_prob >= 0.5, 1, -1)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, pos_label=1))
            rec = float(recall_score(y_test, y_pred, pos_label=1))
            f1 = float(f1_score(y_test, y_pred, pos_label=1))
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            }

            # Save weights
            trained_path = os.path.join(MODELS_DIR, f"trained_keras_{training_id}.weights.h5")
            model.save_weights(trained_path)

        else:
            # Traditional ML using scikit-learn
            clf = None
            hp = model_meta.get("hyperparams", {})
            if model_meta["type"] == "logistic_regression":
                clf = LogisticRegression(C=hp.get("C", 1.0), solver=hp.get("solver", "liblinear"), max_iter=hp.get("max_iter", 100))
            elif model_meta["type"] == "svm":
                clf = SVC(C=hp.get("C", 1.0), kernel=hp.get("kernel", "rbf"))
            elif model_meta["type"] == "random_forest":
                clf = RandomForestClassifier(n_estimators=hp.get("n_estimators", 100), max_depth=hp.get("max_depth", None), random_state=42)
            elif model_meta["type"] == "knn":
                clf = KNeighborsClassifier(n_neighbors=hp.get("n_neighbors", 5))
            elif model_meta["type"] == "naive_bayes":
                clf = GaussianNB()
            else:
                continue

            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, pos_label=1))
            rec = float(recall_score(y_test, y_pred, pos_label=1))
            f1 = float(f1_score(y_test, y_pred, pos_label=1))
            cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            }

            trained_path = os.path.join(MODELS_DIR, f"trained_sklearn_{training_id}.pkl")
            with open(trained_path, "wb") as mf:
                pickle.dump(clf, mf)

        # Persist training record
        # Collect params for logging to config
        params_record = {}
        if model_meta["type"] == "keras_mlp":
            try:
                first_dense = next((ld["Dense"] for ld in architecture["layers"] if "Dense" in ld), {})
                hidden_units_used = int(first_dense.get("units", 8))
            except Exception:
                hidden_units_used = None
            params_record = {"optimizer": optimizer, "loss": loss, "epochs": epochs, "batch_size": batch_size, "hidden_units": hidden_units_used}

        cfg["trainings"].append({
            "id": training_id,
            "dataset_id": dataset_id,
            "model_id": mid,
            "metrics": metrics,
            "trained_path": os.path.relpath(trained_path, BASE_DIR).replace("\\", "/") if trained_path else None,
            "preprocess_path": os.path.relpath(preprocess_path, BASE_DIR).replace("\\", "/"),
            "history_path": os.path.relpath(history_path, BASE_DIR).replace("\\", "/") if history_path else None,
            "params": params_record,
            "created_at": datetime.utcnow().isoformat()
        })

        results.append({
            "model": model_meta,
            "metrics": metrics
        })

    save_config(cfg)
    flash("Training complete.")
    return redirect(url_for("results", dataset_id=dataset_id))


@app.route("/results")
def results():
    ensure_dirs()
    cfg = load_config()
    dataset_id = request.args.get("dataset_id")
    trainings = cfg.get("trainings", [])
    datasets = cfg.get("datasets", [])
    models = {m["id"]: m for m in cfg.get("models", [])}

    if dataset_id:
        ds_trainings = [t for t in trainings if t["dataset_id"] == dataset_id]
    else:
        ds_trainings = trainings

    # Group latest training per model for this dataset
    latest_by_model = {}
    for t in ds_trainings:
        latest_by_model[t["model_id"]] = t

    latest_list = [
        {
            "model": models.get(mid),
            "training": tr
        }
        for mid, tr in latest_by_model.items()
        if mid in models
    ]

    return render_template("results.html", dataset_id=dataset_id, datasets=datasets, results=latest_list)


@app.route("/training/<training_id>/delete", methods=["POST"])
def training_delete(training_id):
    """Delete a specific training record and associated files, then redirect back to Workspace filtered by its dataset."""
    ensure_dirs()
    cfg = load_config()
    trainings = cfg.get("trainings", [])
    t = next((tr for tr in trainings if tr.get("id") == training_id), None)
    if not t:
        flash("Training not found.")
        return redirect(url_for("workspace"))

    # Remove files if present
    for key in ("trained_path", "preprocess_path", "history_path"):
        p = t.get(key)
        if not p:
            continue
        p_abs = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
        try:
            if os.path.exists(p_abs):
                os.remove(p_abs)
        except Exception:
            # Ignore file removal errors
            pass

    # Remove training from config
    cfg["trainings"] = [tr for tr in trainings if tr.get("id") != training_id]
    save_config(cfg)
    flash("Training deleted.")
    ds_id = t.get("dataset_id")
    return redirect(url_for("workspace", dataset_id=ds_id))


@app.route("/api/metrics/<dataset_id>")
def api_metrics(dataset_id):
    cfg = load_config()
    trainings = [t for t in cfg.get("trainings", []) if t["dataset_id"] == dataset_id]
    models = {m["id"]: m for m in cfg.get("models", [])}
    latest = {}
    for t in trainings:
        latest[t["model_id"]] = t
    payload = []
    for mid, t in latest.items():
        m = models.get(mid)
        if not m:
            continue
        payload.append({
            "model_name": m["name"],
            "model_type": m["type"],
            "f1": t["metrics"]["f1"],
            "accuracy": t["metrics"]["accuracy"]
        })
    return jsonify({"metrics": payload})

@app.route("/api/history/<training_id>")
def api_history(training_id):
    cfg = load_config()
    tr = next((t for t in cfg.get("trainings", []) if t["id"] == training_id), None)
    if not tr or not tr.get("history_path"):
        return jsonify({"error": "history_not_found"}), 404
    hist_path = tr["history_path"]
    hist_abs = os.path.join(BASE_DIR, hist_path) if not os.path.isabs(hist_path) else hist_path
    if not os.path.exists(hist_abs):
        return jsonify({"error": "history_file_missing"}), 404
    with open(hist_abs, "r", encoding="utf-8") as hf:
        try:
            data = json.load(hf)
        except Exception:
            return jsonify({"error": "history_read_error"}), 500
    return jsonify(data)


# Async Training API
def _optimizer_from_name(name, lr):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return None
    name = (name or "Adam").lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    if name == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=lr)
    if name == "adamax":
        return keras.optimizers.Adamax(learning_rate=lr)
    return keras.optimizers.Adam(learning_rate=lr)


def _run_training_job(job_id, dataset_id, model_ids, params):
    global TRAIN_JOB
    TRAIN_JOB.update({
        "id": job_id,
        "status": "running",
        "progress": 0.0,
        "current_index": 0,
        "total": len(model_ids),
        "messages": ["Training started"],
        "stop": False,
        "error": None
    })
    try:
        ensure_dirs()
        cfg = load_config()
        datasets = cfg.get("datasets", [])
        models = cfg.get("models", [])
        trainings = cfg.get("trainings", [])
        ds_meta = next((d for d in datasets if d["id"] == dataset_id), None)
        if not ds_meta:
            raise RuntimeError("Dataset not found")

        dataset_path = os.path.join(BASE_DIR, ds_meta["path"]) if not os.path.isabs(ds_meta["path"]) else ds_meta["path"]
        df = pd.read_csv(dataset_path)
        X = df[ds_meta["features"]].values.astype(float)
        y = df[ds_meta["target"]].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Honor pre-scaling from preprocessing wizard
        preproc = ds_meta.get("preprocessing", {}) if ds_meta else {}
        pre_scaled = bool(preproc.get("scaling"))
        if pre_scaled:
            scaler = IdentityScaler()
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        for idx, mid in enumerate(model_ids):
            if TRAIN_JOB.get("stop"):
                TRAIN_JOB["messages"].append("Training interrupted")
                TRAIN_JOB["status"] = "completed"
                break
            TRAIN_JOB["current_index"] = idx
            TRAIN_JOB["messages"].append(f"Training model {idx+1}/{len(model_ids)}")
            TRAIN_JOB["epoch"] = 0
            TRAIN_JOB["epoch_total"] = 0
            model_meta = next((m for m in models if m["id"] == mid), None)
            if not model_meta:
                TRAIN_JOB["messages"].append("Model not found; skipping")
                continue
            training_id = str(uuid.uuid4())
            preprocess_path = os.path.join(MODELS_DIR, f"preprocess_{training_id}.pkl")
            with open(preprocess_path, "wb") as pf:
                pickle.dump({
                    "scaler": scaler,
                    "feature_columns": ds_meta["features"],
                    "target": ds_meta["target"],
                    "label_map": ds_meta.get("label_map", {}),
                    "pre_scaled": pre_scaled,
                    "scaling_method": preproc.get("scaling")
                }, pf)

            metrics = {}
            history_path = None
            trained_path = None

            if model_meta["type"] == "keras_mlp":
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                except Exception as e:
                    TRAIN_JOB["messages"].append(f"TensorFlow/Keras unavailable: {e}; skipping {model_meta['name']}")
                    continue

                arch_path_rel = model_meta["hyperparams"]["architecture_path"]
                arch_path = os.path.join(BASE_DIR, arch_path_rel) if not os.path.isabs(arch_path_rel) else arch_path_rel
                with open(arch_path, "r", encoding="utf-8") as f:
                    architecture = json.load(f)

                model = keras.Sequential()
                input_dim = X_train_scaled.shape[1]
                for layer_def in architecture["layers"]:
                    if "Dense" in layer_def:
                        params_layer = layer_def["Dense"]
                        units = int(params_layer.get("units", 8))
                        activation = params_layer.get("activation", "relu")
                        if len(model.layers) == 0:
                            model.add(keras.layers.Dense(units=units, activation=activation, input_shape=(input_dim,)))
                        else:
                            model.add(keras.layers.Dense(units=units, activation=activation))
                model.add(keras.layers.Dense(units=1, activation="sigmoid"))

                # Resolve training parameters with preference to model hyperparams
                hp = model_meta.get("hyperparams", {})
                epochs = max(1, int(hp.get("epochs", params.get("epochs", 100))))
                batch_size = max(1, int(hp.get("batch_size", params.get("batch_size", 32))))
                lr = float(hp.get("learning_rate", params.get("learning_rate", 0.001)))
                opt_name = hp.get("optimizer", params.get("optimizer", "Adam"))
                loss_fn = hp.get("loss", params.get("loss", "binary_crossentropy"))
                opt = _optimizer_from_name(opt_name, lr)
                TRAIN_JOB["messages"].append(
                    f"Using Keras MLP params: optimizer={opt_name}, loss={loss_fn}, epochs={epochs}, batch_size={batch_size}"
                )
                if opt is None:
                    TRAIN_JOB["messages"].append("Keras not available; skipping")
                    continue
                model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"]) 
                TRAIN_JOB["epoch_total"] = epochs

                # Keras targets {0,1}
                y_train_01 = ((y_train + 1) // 2).astype(int)
                y_test_01 = ((y_test + 1) // 2).astype(int)

                class InterruptCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        TRAIN_JOB["progress"] = (idx + (epoch + 1) / epochs) / len(model_ids)
                        TRAIN_JOB["epoch"] = epoch + 1
                        if TRAIN_JOB.get("stop"):
                            self.model.stop_training = True
                            TRAIN_JOB["messages"].append("Requested stop; halting current model...")

                history = model.fit(
                    X_train_scaled, y_train_01,
                    validation_data=(X_test_scaled, y_test_01),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[InterruptCallback()]
                )

                history_path = os.path.join(MODELS_DIR, f"history_{training_id}.json")
                with open(history_path, "w", encoding="utf-8") as hf:
                    json.dump({
                        "loss": history.history.get("loss", []),
                        "val_loss": history.history.get("val_loss", [])
                    }, hf, indent=2)

                y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
                y_pred = np.where(y_pred_prob >= 0.5, 1, -1)

                acc = float(accuracy_score(y_test, y_pred))
                prec_per = precision_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                rec_per = recall_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                f1_per = f1_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
                f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

                metrics = {
                    "accuracy": acc,
                    "accuracy_percent": acc * 100.0,
                    "precision_macro": prec_macro,
                    "recall_macro": rec_macro,
                    "f1": f1_macro,
                    "f1_macro": f1_macro,
                    "precision_per_class": {"-1": float(prec_per[0]), "1": float(prec_per[1])},
                    "recall_per_class": {"-1": float(rec_per[0]), "1": float(rec_per[1])},
                    "f1_per_class": {"-1": float(f1_per[0]), "1": float(f1_per[1])},
                    "confusion": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
                }

                trained_path = os.path.join(MODELS_DIR, f"trained_keras_{training_id}.weights.h5")
                model.save_weights(trained_path)

            else:
                clf = None
                hp = model_meta.get("hyperparams", {})
                if model_meta["type"] == "logistic_regression":
                    clf = LogisticRegression(C=hp.get("C", 1.0), solver=hp.get("solver", "liblinear"), max_iter=hp.get("max_iter", 100))
                elif model_meta["type"] == "svm":
                    clf = SVC(C=hp.get("C", 1.0), kernel=hp.get("kernel", "rbf"))
                elif model_meta["type"] == "random_forest":
                    clf = RandomForestClassifier(n_estimators=hp.get("n_estimators", 100), max_depth=hp.get("max_depth", None), random_state=42)
                elif model_meta["type"] == "knn":
                    clf = KNeighborsClassifier(n_neighbors=hp.get("n_neighbors", 5))
                elif model_meta["type"] == "naive_bayes":
                    clf = GaussianNB()
                else:
                    TRAIN_JOB["messages"].append("Unsupported model; skipping")
                    continue

                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)

                acc = float(accuracy_score(y_test, y_pred))
                prec_per = precision_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                rec_per = recall_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                f1_per = f1_score(y_test, y_pred, labels=[-1, 1], average=None, zero_division=0)
                prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
                f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

                metrics = {
                    "accuracy": acc,
                    "accuracy_percent": acc * 100.0,
                    "precision_macro": prec_macro,
                    "recall_macro": rec_macro,
                    "f1": f1_macro,
                    "f1_macro": f1_macro,
                    "precision_per_class": {"-1": float(prec_per[0]), "1": float(prec_per[1])},
                    "recall_per_class": {"-1": float(rec_per[0]), "1": float(rec_per[1])},
                    "f1_per_class": {"-1": float(f1_per[0]), "1": float(f1_per[1])},
                    "confusion": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
                }

                trained_path = os.path.join(MODELS_DIR, f"trained_sklearn_{training_id}.pkl")
                with open(trained_path, "wb") as mf:
                    pickle.dump(clf, mf)

            # Prepare params record for logging
            params_record = {}
            if model_meta["type"] == "keras_mlp":
                try:
                    first_dense = next((ld["Dense"] for ld in architecture["layers"] if "Dense" in ld), {})
                    hidden_units_used = int(first_dense.get("units", 8))
                except Exception:
                    hidden_units_used = None
                # Persist the resolved parameters actually used for this training
                params_record = {
                    "optimizer": opt_name,
                    "loss": loss_fn,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "hidden_units": hidden_units_used
                }

            cfg["trainings"].append({
                "id": training_id,
                "dataset_id": dataset_id,
                "model_id": mid,
                "metrics": metrics,
                "trained_path": os.path.relpath(trained_path, BASE_DIR).replace("\\", "/") if trained_path else None,
                "preprocess_path": os.path.relpath(preprocess_path, BASE_DIR).replace("\\", "/"),
                "history_path": os.path.relpath(history_path, BASE_DIR).replace("\\", "/") if history_path else None,
                "params": params_record,
                "created_at": datetime.utcnow().isoformat()
            })

            save_config(cfg)
            TRAIN_JOB["progress"] = (idx + 1) / len(model_ids)
            TRAIN_JOB["messages"].append(f"Finished model {idx+1}/{len(model_ids)}")

        if TRAIN_JOB["status"] == "running":
            TRAIN_JOB["status"] = "completed"
            TRAIN_JOB["messages"].append("Training completed")

    except Exception as e:
        TRAIN_JOB["status"] = "error"
        TRAIN_JOB["error"] = str(e)
        TRAIN_JOB["messages"].append(f"Error: {e}")


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global TRAIN_JOB
    if TRAIN_JOB.get("status") == "running":
        return jsonify({"error": "job_running"}), 409
    data = request.get_json(force=True) or {}
    dataset_id = data.get("dataset_id")
    model_ids = data.get("model_ids") or []
    params = {
        "epochs": max(1, int(data.get("epochs", 50))),
        "batch_size": max(1, int(data.get("batch_size", 32))),
        "learning_rate": float(data.get("learning_rate", 0.001)),
        "optimizer": data.get("optimizer", "Adam"),
        "loss": data.get("loss", "binary_crossentropy"),
    }
    if not dataset_id or not model_ids:
        return jsonify({"error": "invalid_request"}), 400
    job_id = str(uuid.uuid4())
    t = threading.Thread(target=_run_training_job, args=(job_id, dataset_id, model_ids, params), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/train/status")
def api_train_status():
    return jsonify({
        "id": TRAIN_JOB.get("id"),
        "status": TRAIN_JOB.get("status"),
        "progress": TRAIN_JOB.get("progress"),
        "current_index": TRAIN_JOB.get("current_index"),
        "total": TRAIN_JOB.get("total"),
        "messages": TRAIN_JOB.get("messages", []),
        "error": TRAIN_JOB.get("error"),
        "epoch": TRAIN_JOB.get("epoch"),
        "epoch_total": TRAIN_JOB.get("epoch_total")
    })


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    TRAIN_JOB["stop"] = True
    return jsonify({"stopping": True})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    ensure_dirs()
    cfg = load_config()
    models = cfg.get("models", [])
    trainings = cfg.get("trainings", [])

    latest_train_by_model = {}
    for t in trainings:
        latest_train_by_model[t["model_id"]] = t

    selected_model_id = request.args.get("model_id") if request.method == "GET" else request.form.get("model_id")
    selected_training = latest_train_by_model.get(selected_model_id) if selected_model_id else None
    feature_columns = []
    if selected_training:
        import pickle
        pp_path = os.path.join(BASE_DIR, selected_training["preprocess_path"]) if not os.path.isabs(selected_training["preprocess_path"]) else selected_training["preprocess_path"]
        try:
            with open(pp_path, "rb") as pf:
                pp = pickle.load(pf)
                feature_columns = pp.get("feature_columns", [])
        except Exception:
            feature_columns = []

    if request.method == "GET":
        return render_template("predict.html", models=models, feature_columns=feature_columns, selected_model_id=selected_model_id)

    # POST: perform prediction
    if not selected_training or not feature_columns:
        flash("Please select a trained model.")
        return redirect(url_for("predict"))

    # Collect feature values in correct order
    try:
        X_input = [float(request.form.get(f"feature_{col}")) for col in feature_columns]
    except Exception:
        flash("Please enter numeric values for all features.")
        return redirect(url_for("predict", model_id=selected_model_id))

    import numpy as np
    import pickle
    pp_path = os.path.join(BASE_DIR, selected_training["preprocess_path"]) if not os.path.isabs(selected_training["preprocess_path"]) else selected_training["preprocess_path"]
    with open(pp_path, "rb") as pf:
        pp = pickle.load(pf)
    scaler = pp["scaler"]
    X_scaled = scaler.transform(np.array([X_input]))

    # Load model and predict
    model_meta = next((m for m in models if m["id"] == selected_model_id), None)
    pred_class = None
    if model_meta["type"] == "keras_mlp":
        try:
            import tensorflow as tf
            from tensorflow import keras
        except Exception as e:
            flash(f"TensorFlow/Keras not available: {e}.")
            return redirect(url_for("predict"))
        # Load architecture
        arch_path_rel = model_meta["hyperparams"]["architecture_path"]
        arch_path = os.path.join(BASE_DIR, arch_path_rel) if not os.path.isabs(arch_path_rel) else arch_path_rel
        with open(arch_path, "r", encoding="utf-8") as f:
            architecture = json.load(f)
        model = keras.Sequential()
        input_dim = X_scaled.shape[1]
        for layer_def in architecture["layers"]:
            if "Dense" in layer_def:
                params = layer_def["Dense"]
                units = int(params.get("units", 8))
                activation = params.get("activation", "relu")
                if len(model.layers) == 0:
                    model.add(keras.layers.Dense(units=units, activation=activation, input_shape=(input_dim,)))
                else:
                    model.add(keras.layers.Dense(units=units, activation=activation))

        # Compile consistent with training defaults
        model.compile(optimizer=model_meta["hyperparams"].get("optimizer", "Adam"), loss=model_meta["hyperparams"].get("loss", "binary_crossentropy"))
        weights_path = os.path.join(BASE_DIR, selected_training["trained_path"]) if not os.path.isabs(selected_training["trained_path"]) else selected_training["trained_path"]
        # Backward-compat: normalize legacy .h5 weights to .weights.h5 for Keras 3
        if weights_path.endswith(".h5") and not weights_path.endswith(".weights.h5"):
            alt_path = weights_path[:-3] + ".weights.h5"
            try:
                if os.path.exists(weights_path) and not os.path.exists(alt_path):
                    os.rename(weights_path, alt_path)
                cfg = load_config()
                for t in cfg.get("trainings", []):
                    if t.get("id") == selected_training.get("id"):
                        t["trained_path"] = os.path.relpath(alt_path, BASE_DIR)
                        save_config(cfg)
                        selected_training["trained_path"] = t["trained_path"]
                        break
                weights_path = alt_path
            except Exception:
                weights_path = alt_path
        model.load_weights(weights_path)
        prob = float(model.predict(X_scaled, verbose=0).ravel()[0])
        pred_class = 1 if prob >= 0.5 else -1
    else:
        import pickle
        mdl_path = os.path.join(BASE_DIR, selected_training["trained_path"]) if not os.path.isabs(selected_training["trained_path"]) else selected_training["trained_path"]
        with open(mdl_path, "rb") as mf:
            clf = pickle.load(mf)
        pred_class = int(clf.predict(X_scaled)[0])

    result_text = "Class B (+1)" if pred_class == 1 else "Class A (-1)"
    flash(f"Prediction: {result_text}")
    return redirect(url_for("predict", model_id=selected_model_id))


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=True)