#!/usr/bin/env python3
"""
Train a Random Forest to predict bottleneck size (Nb) from the allele-frequency
spectrum of shared mutations (circles + duplex), and optionally apply the trained
model to empirical data.

Usage examples
--------------
# Train with defaults
python rf.py train aggregated_shared_mutations.csv

# Train with alt-count filtering and log target
python rf.py train aggregated_shared_mutations.csv --min-circles-alt 2 --min-duplex-alt 2 --log-target

# Hyperparameter search
python rf.py train aggregated_shared_mutations.csv --tune --n-iter 100

# Predict Nb from empirical data using a trained model
python rf.py predict empirical_shared.csv --model-dir model_output

# Predict with same alt filters used during training
python rf.py predict empirical_shared.csv --model-dir model_output --min-circles-alt 2 --min-duplex-alt 2
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_af_bins(n_bins=20):
    """Log-spaced bins from 1e-4 to 1."""
    return np.logspace(-4, 0, n_bins + 1)


def compute_af_features(df, bins):
    """
    Turn a dataframe of shared mutations into a single feature vector:
    histogram(circles AF) + histogram(duplex AF).

    Works for both simulation param sets and empirical data.
    Returns a 1D array of length 2 * (len(bins) - 1).
    """
    n_bins = len(bins) - 1
    if len(df) == 0:
        return np.zeros(2 * n_bins)

    caf = df["circles_alt"] / df["circles_depth"]
    daf = df["duplex_alt"] / df["duplex_depth"]

    c_hist, _ = np.histogram(caf, bins=bins)
    d_hist, _ = np.histogram(daf, bins=bins)

    return np.concatenate([c_hist, d_hist])


def filter_by_alt(df, min_circles_alt, min_duplex_alt):
    """Drop mutations below alt-count thresholds."""
    if min_circles_alt <= 0 and min_duplex_alt <= 0:
        return df
    n_before = len(df)
    df = df[
        (df["circles_alt"] >= min_circles_alt) & (df["duplex_alt"] >= min_duplex_alt)
    ].copy()
    print(
        f"  Alt filter (circles>={min_circles_alt}, duplex>={min_duplex_alt}): "
        f"kept {len(df):,} / {n_before:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Training data prep
# ---------------------------------------------------------------------------


def load_training_data(csv_path, min_circles_alt=0, min_duplex_alt=0):
    """Load the aggregated simulation CSV, expand replicates, apply filters."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} mutations from {csv_path}")
    print(f"  Unique param sets: {df['param_set'].nunique()}")

    # each (param_set, run) combo is its own training sample
    df["param_set"] = df["param_set"].astype(str) + "_rep" + df["run"].astype(str)
    print(f"  After replicate expansion: {df['param_set'].nunique()} samples")

    # save metadata before filtering so zero-mutation samples are preserved
    meta = (
        df.groupby("param_set").first()[["Nb", "N", "u_per_site", "Gs"]].reset_index()
    )

    df = filter_by_alt(df, min_circles_alt, min_duplex_alt)
    n_empty = len(meta) - df["param_set"].nunique()
    if n_empty > 0:
        print(f"  {n_empty} samples now have 0 mutations (still used as zero vectors)")

    return df, meta


def build_feature_matrix(df, bins, all_meta):
    """
    One row per param_set. Samples with no mutations after filtering get an
    all-zero feature vector -- that's real information for the model.
    """
    grouped = df.groupby("param_set") if len(df) > 0 else None
    has_data = set(grouped.groups.keys()) if grouped is not None else set()

    all_ids = list(all_meta["param_set"])
    meta_lookup = all_meta.set_index("param_set").to_dict("index")

    n_features = 2 * (len(bins) - 1)
    X = np.zeros((len(all_ids), n_features))
    y = np.zeros(len(all_ids))
    rows = []

    for i, pid in enumerate(all_ids):
        if pid in has_data:
            g = grouped.get_group(pid)
            X[i] = compute_af_features(g, bins)
            n_mut = len(g)
        else:
            n_mut = 0

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
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  {n_with} with mutations, {len(all_ids) - n_with} zero-vector samples")
    print(f"  Nb range: [{y.min():.1f}, {y.max():.1f}]")

    return X, y, np.array(all_ids), metadata


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_hyperparameters(X_train, y_train, n_iter=100, cv=5):
    """Randomized search. Scores on neg MSE (more stable than R^2 for CV)."""
    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER SEARCH ({n_iter} iters, {cv}-fold CV)")
    print(f"{'=' * 60}")

    param_space = {
        "n_estimators": randint(100, 800),
        "max_depth": [None, 10, 20, 30, 50, 75],
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 15),
        "max_features": [0.2, 0.3, 0.5, 0.7, "sqrt", "log2", None],
        "bootstrap": [True, False],
        "max_samples": [None, 0.6, 0.7, 0.8, 0.9],
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
    res["cv_rmse"] = np.sqrt(-res["mean_test_score"])
    res["train_rmse"] = np.sqrt(-res["mean_train_score"])

    print(f"\nBest CV RMSE: {res['cv_rmse'].iloc[0]:.3f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    print("\nTop 5:")
    show = ["cv_rmse", "train_rmse"] + [
        c for c in res.columns if c.startswith("param_")
    ]
    show = [c for c in show if c in res.columns]
    print(res[show].head(5).to_string(index=False))

    return search.best_estimator_, search, res


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(X_train, y_train, X_test, y_test, **rf_kwargs):
    """Fit RF, report train/test/CV metrics."""
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")
    for k, v in rf_kwargs.items():
        print(f"  {k}: {v}")

    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_kwargs)
    rf.fit(X_train, y_train)

    metrics = {}
    for name, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = rf.predict(Xs)
        metrics[f"{name}_r2"] = r2_score(ys, pred)
        metrics[f"{name}_rmse"] = np.sqrt(mean_squared_error(ys, pred))
        metrics[f"{name}_mae"] = mean_absolute_error(ys, pred)
        metrics[f"y_{name}"] = ys
        metrics[f"y_{name}_pred"] = pred

    for name in ["train", "test"]:
        print(
            f"  {name.upper()}: R²={metrics[f'{name}_r2']:.4f}  "
            f"RMSE={metrics[f'{name}_rmse']:.2f}  MAE={metrics[f'{name}_mae']:.2f}"
        )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    metrics["cv_r2_mean"] = cv_scores.mean()
    metrics["cv_r2_std"] = cv_scores.std()
    print(f"  5-fold CV R²: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return rf, metrics


def get_feature_importance(rf, bins, top_n=10):
    """Rank AF bins by importance."""
    n_bins = len(bins) - 1
    imp = rf.feature_importances_
    names = [f"circles_bin{i}" for i in range(n_bins)] + [
        f"duplex_bin{i}" for i in range(n_bins)
    ]
    af_ranges = [f"[{bins[i]:.1e}, {bins[i + 1]:.1e})" for i in range(n_bins)] * 2

    importance_df = pd.DataFrame(
        {"feature": names, "importance": imp, "af_range": af_ranges}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features:")
    print(importance_df.head(top_n).to_string(index=False))
    return importance_df


# ---------------------------------------------------------------------------
# Empirical prediction
# ---------------------------------------------------------------------------


def predict_empirical(
    model, bins, empirical_csv, min_circles_alt=0, min_duplex_alt=0, log_target=False
):
    """
    Predict Nb from empirical shared-mutation data.

    Uses individual tree predictions to get uncertainty estimates -- the spread
    across trees in the forest acts like a built-in ensemble confidence interval.
    """

    df = pd.read_csv(empirical_csv)
    print(f"Loaded {len(df)} shared mutations from {empirical_csv}")

    required = ["circles_depth", "circles_alt", "duplex_depth", "duplex_alt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = filter_by_alt(df, min_circles_alt, min_duplex_alt)

    X = compute_af_features(df, bins).reshape(1, -1)
    print(f"  {int(X.sum())} mutations binned into {X.shape[1]} features")

    # each tree's prediction gives us a distribution
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])

    if log_target:
        tree_preds = 10**tree_preds

    point = np.mean(tree_preds)
    median = np.median(tree_preds)
    std = np.std(tree_preds)
    ci95 = (np.percentile(tree_preds, 2.5), np.percentile(tree_preds, 97.5))
    ci68 = (np.percentile(tree_preds, 16), np.percentile(tree_preds, 84))
    iqr = (np.percentile(tree_preds, 25), np.percentile(tree_preds, 75))

    print(f"\n  Mean:    {point:.1f}")
    print(f"  Median:  {median:.1f}")
    print(f"  Std:     {std:.1f}")
    print(f"  IQR:     [{iqr[0]:.1f}, {iqr[1]:.1f}]")
    print(f"  68% CI:  [{ci68[0]:.1f}, {ci68[1]:.1f}]")
    print(f"  95% CI:  [{ci95[0]:.1f}, {ci95[1]:.1f}]")
    print(f"  ({model.n_estimators} trees)")

    return {
        "nb_mean": point,
        "nb_median": median,
        "nb_std": std,
        "nb_lower_95": ci95[0],
        "nb_upper_95": ci95[1],
        "nb_lower_68": ci68[0],
        "nb_upper_68": ci68[1],
        "nb_iqr_lower": iqr[0],
        "nb_iqr_upper": iqr[1],
        "n_mutations_used": len(df),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_training_outputs(
    output_dir,
    rf,
    bins,
    metrics,
    metadata,
    importance_df,
    search_results=None,
    log_target=False,
):
    """Write model and all artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, out / "model.joblib")
    np.save(out / "af_bins.npy", bins)

    # save config so predict knows how the model was trained
    pd.DataFrame([{"log_target": log_target}]).to_csv(out / "config.csv", index=False)

    scalar = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    pd.DataFrame([scalar]).to_csv(out / "metrics.csv", index=False)

    hparams = {
        "n_estimators": rf.n_estimators,
        "max_depth": rf.max_depth,
        "min_samples_split": rf.min_samples_split,
        "min_samples_leaf": rf.min_samples_leaf,
        "max_features": rf.max_features,
        "bootstrap": rf.bootstrap,
    }
    pd.DataFrame([hparams]).to_csv(out / "hyperparameters.csv", index=False)

    pd.DataFrame(
        {"y_true": metrics["y_test"], "y_pred": metrics["y_test_pred"]}
    ).to_csv(out / "test_predictions.csv", index=False)

    importance_df.to_csv(out / "feature_importance.csv", index=False)
    metadata.to_csv(out / "metadata.csv", index=False)

    if search_results is not None:
        search_results.to_csv(out / "search_results.csv", index=False)

    print(f"\nAll outputs saved to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train RF to predict Nb, or predict Nb from empirical data"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    tp = sub.add_parser("train", help="Train model on simulation data")
    tp.add_argument("input_csv", type=Path)
    tp.add_argument("-o", "--output-dir", type=Path, default="model_output")
    tp.add_argument("--n-bins", type=int, default=20)
    tp.add_argument("--test-size", type=float, default=0.2)
    tp.add_argument("--min-circles-alt", type=int, default=5)
    tp.add_argument("--min-duplex-alt", type=int, default=3)
    tp.add_argument(
        "--log-target", action="store_true", help="Predict log10(Nb) instead of raw Nb"
    )
    tp.add_argument("--tune", action="store_true")
    tp.add_argument("--n-iter", type=int, default=100)
    tp.add_argument("--cv", type=int, default=5)
    tp.add_argument("--n-estimators", type=int, default=200)
    tp.add_argument("--max-depth", type=int, default=None)
    tp.add_argument("--min-samples-split", type=int, default=2)
    tp.add_argument("--min-samples-leaf", type=int, default=1)
    tp.add_argument("--max-features", type=str, default="sqrt")
    tp.add_argument("--no-bootstrap", action="store_true")

    # --- predict ---
    pp = sub.add_parser("predict", help="Predict Nb from empirical data")
    pp.add_argument("empirical_csv", type=Path)
    pp.add_argument("--model-dir", type=Path, default="model_output")
    pp.add_argument("--min-circles-alt", type=int, default=0)
    pp.add_argument("--min-duplex-alt", type=int, default=0)
    pp.add_argument("-o", "--output", type=Path, default=None)

    return parser


def cmd_train(args):
    df, all_meta = load_training_data(
        args.input_csv, args.min_circles_alt, args.min_duplex_alt
    )

    bins = make_af_bins(args.n_bins)
    X, y, _, metadata = build_feature_matrix(df, bins, all_meta)

    if args.log_target:
        y = np.log10(y)
        print(f"  log10(Nb) range: [{y.min():.2f}, {y.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    search_results = None
    if args.tune:
        rf, _, search_results = tune_hyperparameters(
            X_train, y_train, n_iter=args.n_iter, cv=args.cv
        )
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
    importance_df = get_feature_importance(rf, bins)

    save_training_outputs(
        args.output_dir,
        rf,
        bins,
        metrics,
        metadata,
        importance_df,
        search_results,
        log_target=args.log_target,
    )
    print("\nDone.")


def cmd_predict(args):
    model_dir = Path(args.model_dir)
    rf = joblib.load(model_dir / "model.joblib")
    bins = np.load(model_dir / "af_bins.npy")

    # check if model was trained on log scale
    log_target = False
    config_path = model_dir / "config.csv"
    if config_path.exists():
        log_target = bool(pd.read_csv(config_path)["log_target"].iloc[0])
        if log_target:
            print("  Note: model was trained on log10(Nb)")

    results = predict_empirical(
        rf,
        bins,
        args.empirical_csv,
        min_circles_alt=args.min_circles_alt,
        min_duplex_alt=args.min_duplex_alt,
        log_target=log_target,
    )

    if args.output:
        pd.DataFrame([results]).to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
