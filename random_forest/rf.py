#!/usr/bin/env python3
"""
Train a Random Forest to predict bottleneck size (Nb) from the allele-frequency
spectrum of shared mutations (circles + duplex).

Usage examples
--------------
# Basic run with defaults
python rf.py aggregated_shared_mutations.csv

# With alt-count filtering (keeps zero-mutation param sets as zero vectors)
python rf.py aggregated_shared_mutations.csv --min-circles-alt 2 --min-duplex-alt 2

# Hyperparameter search
python rf.py aggregated_shared_mutations.csv --tune --n-iter 100

# Predict on log scale
python rf.py aggregated_shared_mutations.csv --log-target
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint, uniform


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_and_prepare(csv_path, min_circles_alt=0, min_duplex_alt=0):
    """Load the aggregated CSV and expand replicates into separate samples."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} mutations from {csv_path}")
    print(f"  Unique param sets: {df['param_set'].nunique()}")

    # Each (param_set, run) combo becomes its own sample
    df["param_set"] = df["param_set"].astype(str) + "_rep" + df["run"].astype(str)
    print(f"  After replicate expansion: {df['param_set'].nunique()} samples")

    # Grab metadata for every sample BEFORE filtering mutations
    meta = (
        df.groupby("param_set").first()[["Nb", "N", "u_per_site", "Gs"]].reset_index()
    )

    # Filter by alt counts
    if min_circles_alt > 0 or min_duplex_alt > 0:
        n_before = len(df)
        df = df[
            (df["circles_alt"] >= min_circles_alt)
            & (df["duplex_alt"] >= min_duplex_alt)
        ]
        n_removed = n_before - len(df)
        n_empty = len(meta) - df["param_set"].nunique()
        print(
            f"  Alt filter (circles>={min_circles_alt}, duplex>={min_duplex_alt}): "
            f"removed {n_removed:,} mutations, {n_empty} samples now have 0 mutations"
        )

    return df, meta


def make_af_bins(n_bins=20):
    """Log-spaced bins from 1e-4 to 1."""
    return np.logspace(-4, 0, n_bins + 1)


def build_feature_matrix(df, bins, all_meta):
    """
    One row per param_set: histogram of circles AF + histogram of duplex AF.
    Param sets with no mutations (post-filter) get an all-zero feature vector,
    which is genuinely informative for the model.
    """
    n_bins = len(bins) - 1
    grouped = df.groupby("param_set")
    has_data = set(grouped.groups.keys())

    all_ids = list(all_meta["param_set"])
    meta_lookup = all_meta.set_index("param_set").to_dict("index")

    X = np.zeros((len(all_ids), 2 * n_bins))
    y = np.zeros(len(all_ids))
    rows = []

    for i, pid in enumerate(all_ids):
        if pid in has_data:
            g = grouped.get_group(pid)
            caf = g["circles_alt"] / g["circles_depth"]
            daf = g["duplex_alt"] / g["duplex_depth"]
            c_counts, _ = np.histogram(caf, bins=bins)
            d_counts, _ = np.histogram(daf, bins=bins)
            X[i] = np.concatenate([c_counts, d_counts])
            n_mut = len(g)
        else:
            n_mut = 0  # X[i] already zeros

        m = meta_lookup[pid]
        y[i] = m["Nb"]
        rows.append(
            {
                "param_set": pid,
                "Nb": m["Nb"],
                "N": m["N"],
                "u_per_site": m["u_per_site"],
                "Gs": m["Gs"],
                "n_shared_mutations": n_mut,
            }
        )

    metadata = pd.DataFrame(rows)

    n_with = sum(1 for p in all_ids if p in has_data)
    n_without = len(all_ids) - n_with
    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  {n_with} with mutations, {n_without} with zero mutations")
    print(f"  Nb range: [{y.min():.1f}, {y.max():.1f}]")

    return X, y, np.array(all_ids), metadata


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_hyperparameters(X_train, y_train, n_iter=100, cv=5):
    """
    Randomized search over a broad parameter space.

    Uses negative MSE as scoring (more stable than R² for CV), then reports
    both R² and RMSE for interpretability.
    """

    param_space = {
        "n_estimators": randint(100, 800),
        "max_depth": [None, 10, 20, 30, 50, 75],
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 15),
        "max_features": [0.2, 0.3, 0.5, 0.7, "sqrt", "log2", None],
        "bootstrap": [True, False],
        "max_samples": [
            None,
            0.6,
            0.7,
            0.8,
            0.9,
        ],  # subsample ratio when bootstrap=True
    }

    for k, v in param_space.items():
        print(f"  {k}: {v}" if isinstance(v, list) else f"  {k}: {v.args}")

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

    # Convert neg MSE to RMSE for readability
    res["cv_rmse"] = np.sqrt(-res["mean_test_score"])
    res["train_rmse"] = np.sqrt(-res["mean_train_score"])

    print(f"\nBest CV RMSE: {res['cv_rmse'].iloc[0]:.3f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    print("\nTop 5 configs:")
    show_cols = ["cv_rmse", "train_rmse"] + [
        c for c in res.columns if c.startswith("param_")
    ]
    show_cols = [c for c in show_cols if c in res.columns]
    print(res[show_cols].head(5).to_string(index=False))

    return search.best_estimator_, search, res


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(X_train, y_train, X_test, y_test, **rf_kwargs):
    """Fit a RF with given hyperparams, report metrics."""

    for k, v in rf_kwargs.items():
        print(f"  {k}: {v}")

    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_kwargs)
    rf.fit(X_train, y_train)

    sets = {"train": (X_train, y_train), "test": (X_test, y_test)}
    metrics = {}
    for name, (X, y_true) in sets.items():
        pred = rf.predict(X)
        metrics[f"{name}_r2"] = r2_score(y_true, pred)
        metrics[f"{name}_rmse"] = np.sqrt(mean_squared_error(y_true, pred))
        metrics[f"{name}_mae"] = mean_absolute_error(y_true, pred)
        metrics[f"y_{name}"] = y_true
        metrics[f"y_{name}_pred"] = pred

    for name in ["train", "test"]:
        print(
            f"\n  {name.upper()}:  R²={metrics[f'{name}_r2']:.4f}  "
            f"RMSE={metrics[f'{name}_rmse']:.2f}  MAE={metrics[f'{name}_mae']:.2f}"
        )

    # Cross-val on training set for a sanity check
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    metrics["cv_r2_mean"] = cv_scores.mean()
    metrics["cv_r2_std"] = cv_scores.std()
    print(f"\n  5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return rf, metrics


def print_feature_importance(rf, bins, top_n=10):
    """Show which AF bins matter most."""
    n_bins = len(bins) - 1
    imp = rf.feature_importances_

    names = [f"circles_bin{i}" for i in range(n_bins)] + [
        f"duplex_bin{i}" for i in range(n_bins)
    ]
    af_ranges = [f"[{bins[i]:.1e}, {bins[i + 1]:.1e})" for i in range(n_bins)] * 2

    df = pd.DataFrame({"feature": names, "importance": imp, "af_range": af_ranges})
    df = df.sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features:")
    print(df.head(top_n).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def save_outputs(
    output_dir,
    rf,
    bins,
    metrics,
    metadata,
    importance_df,
    y_test,
    y_test_pred,
    search_results=None,
):
    """Dump everything useful to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, out / "model.joblib")
    np.save(out / "af_bins.npy", bins)

    # Scalar metrics
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    pd.DataFrame([scalar]).to_csv(out / "metrics.csv", index=False)

    # Hyperparameters
    hparams = {
        "n_estimators": rf.n_estimators,
        "max_depth": rf.max_depth,
        "min_samples_split": rf.min_samples_split,
        "min_samples_leaf": rf.min_samples_leaf,
        "max_features": rf.max_features,
        "bootstrap": rf.bootstrap,
    }
    pd.DataFrame([hparams]).to_csv(out / "hyperparameters.csv", index=False)

    # Test predictions
    pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred}).to_csv(
        out / "test_predictions.csv", index=False
    )

    # Feature importance
    importance_df.to_csv(out / "feature_importance.csv", index=False)

    # Metadata with predictions
    metadata.to_csv(out / "metadata.csv", index=False)

    if search_results is not None:
        search_results.to_csv(out / "search_results.csv", index=False)

    print(f"\nAll outputs saved to {out}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Predict Nb from allele-frequency spectrum via Random Forest"
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default="model_output")
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)

    # Filtering
    parser.add_argument(
        "--min-circles-alt",
        type=int,
        default=5,
        help="Min circles alt count per mutation",
    )
    parser.add_argument(
        "--min-duplex-alt",
        type=int,
        default=3,
        help="Min duplex alt count per mutation",
    )

    # Target transform
    parser.add_argument(
        "--log-target", action="store_true", help="Predict log10(Nb) instead of Nb"
    )

    # Tuning
    parser.add_argument(
        "--tune", action="store_true", help="Run randomized hyperparameter search"
    )
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--cv", type=int, default=5)

    # Manual hyperparameters (ignored when --tune)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--no-bootstrap", action="store_true")

    args = parser.parse_args()

    # -- Load & prepare -------------------------------------------------
    df, all_meta = load_and_prepare(
        args.input_csv, args.min_circles_alt, args.min_duplex_alt
    )

    bins = make_af_bins(args.n_bins)
    X, y, param_ids, metadata = build_feature_matrix(df, bins, all_meta)

    # Optionally work on log scale (often helps when Nb spans orders of magnitude)
    if args.log_target:
        y = np.log10(y)
        print(f"  Using log10(Nb) as target: [{y.min():.2f}, {y.max():.2f}]")

    # -- Split -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # -- Train -----------------------------------------------------------
    search_results = None

    if args.tune:
        rf, search, search_results = tune_hyperparameters(
            X_train, y_train, n_iter=args.n_iter, cv=args.cv
        )
        # Re-evaluate with the best estimator so we get clean metrics
        rf_kwargs = {
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "min_samples_split": rf.min_samples_split,
            "min_samples_leaf": rf.min_samples_leaf,
            "max_features": rf.max_features,
            "bootstrap": rf.bootstrap,
        }
    else:
        mf = None if args.max_features == "None" else args.max_features
        rf_kwargs = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": mf,
            "bootstrap": not args.no_bootstrap,
        }

    rf, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, **rf_kwargs)

    # -- Feature importance ---------------------------------------------
    importance_df = print_feature_importance(rf, bins)

    # -- Save everything ------------------------------------------------
    save_outputs(
        args.output_dir,
        rf,
        bins,
        metrics,
        metadata,
        importance_df,
        y_test,
        metrics["y_test_pred"],
        search_results,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
