"""Export the existing fitted models as data for the static browser demo.

Run from any directory with the project's Python environment:
    python scripts/export_browser_models.py

This script never trains a model and never reads the source spreadsheet. Only
inference parameters and the existing public model metadata enter the JSON file.
The joblib artifacts must be the trusted, locally trained project artifacts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs" / "assets" / "models.json"
FEATURES = ["湿度", "湿度变化", "温度", "温度变化", "弦编号"]


def export_forest(model):
    """Keep splits and leaf predictions, excluding all training statistics."""
    if not isinstance(model, RandomForestRegressor) or model.n_outputs_ != 1:
        raise ValueError("Expected a fitted single-output RandomForestRegressor.")
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        leaf = tree.children_left == -1
        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": [None if is_leaf else float(threshold)
                          for is_leaf, threshold in zip(leaf, tree.threshold)],
            "value": [float(value) if is_leaf else None
                      for is_leaf, value in zip(leaf, tree.value[:, 0, 0])],
        })
    return {"type": "random_forest", "input_dtype": "float32", "trees": trees}


def export_ridge(model):
    """Preserve the scaler and regression's separate float64 operations."""
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError("Expected a fitted StandardScaler + Ridge pipeline.")
    scaler, ridge = [step for _, step in model.steps]
    if not isinstance(scaler, StandardScaler) or not isinstance(ridge, Ridge):
        raise ValueError("Unexpected ridge pipeline steps.")
    if not scaler.with_mean or not scaler.with_std or ridge.coef_.shape != (5,):
        raise ValueError("Unexpected scaler configuration or regression output.")
    if np.any(scaler.scale_ <= 0):
        raise ValueError("Scaler scales must be positive.")
    return {
        "type": "ridge",
        "input_dtype": "float64",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }


def main():
    bundle = {"format_version": 1, "default": "random_forest", "metadata": {}, "models": {}}
    exporters = {"random_forest": export_forest, "ridge": export_ridge}
    for key, export in exporters.items():
        folder = ROOT / "output" / f"web_{key}"
        model = joblib.load(folder / "model.joblib")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if list(model.feature_names_in_) != FEATURES or model.n_features_in_ != 5:
            raise ValueError(f"Unexpected input features for {key}.")
        if metadata["features"] != FEATURES or metadata["key"] != key:
            raise ValueError(f"Metadata does not match {key}.")
        bundle["models"][key] = export(model)
        bundle["metadata"][key] = metadata

    # Python JSON's float representation round-trips the fitted float64 values.
    content = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(content + "\n", encoding="utf-8")
    print(f"Exported two existing models to {DESTINATION.relative_to(ROOT)} "
          f"({DESTINATION.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
