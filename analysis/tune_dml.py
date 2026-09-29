"""
Optuna tuning for the Stage-A LinearDML nuisance models — PREDICTIVE ONLY.

WHAT IS TUNED: the two first-stage nuisances of the pooled LinearDML secondary
(estimate.run_pooled_dml) — model_y for E[Y|W] and model_t for E[T|W], where
W = region + trip-year dummies. Selection minimizes out-of-fold MSE on the
nuisance's own prediction task, with the fold count taken from refute.DML_CV
so it mirrors LinearDML's cross-fitting partitions.

WHAT IS NEVER TUNED ON: the causal estimate. Selecting nuisances by theta, its
CI, or p-value is post-selection bias; predictive tuning is the sanctioned
practice (Chernozhukov et al. 2018), and Neyman orthogonality means first-stage
error enters theta only at second order. The final stage is OLS — nothing to tune.

EXPECTATION (stated up front): W is ~15 dummy columns, so both nuisances are
essentially region x year cell means — tuned models have limited room to beat
regularized linear. The value here is a transparent, seeded, logged selection
replacing econml's opaque 'auto' (which in econml 0.16 is a CV bake-off between
a default-hyperparameter RandomForest and a weighted LassoCV — only the lasso
alpha is tuned). This is a credibility upgrade, not a power upgrade.

Output: analysis/outputs/dml_nuisance_tuning.json — best family + params + CV
scores per nuisance, alongside the 'auto'-candidate baselines. Consumed by
refute.dml_nuisance_models(tuned=True), which both estimate.py (--tuned-nuisances)
and the DoWhy refutation battery share, so estimator and refuters cannot diverge.

Usage:
    uv run python analysis/tune_dml.py               # trials from analysis/config.yaml
    uv run python analysis/tune_dml.py --trials 25   # quick pass
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import KFold, cross_val_score

try:
    from analysis.config import CFG
    from analysis.estimate import load_frame
    from analysis.refute import DML_CV
except ImportError:
    from config import CFG
    from estimate import load_frame
    from refute import DML_CV

SEED = CFG["seed"]
OUT_PATH = Path(__file__).resolve().parent / "outputs" / "dml_nuisance_tuning.json"


def build_model(family: str, params: dict):
    """One constructor shared by the objective and the consumer (refute.py)."""
    if family == "ridge":
        return Ridge(alpha=params["alpha"], random_state=SEED)
    if family == "random_forest":
        return RandomForestRegressor(random_state=SEED, **params)
    if family == "gradient_boosting":
        return GradientBoostingRegressor(random_state=SEED, **params)
    raise ValueError(f"unknown family {family!r}")


def _suggest(trial: optuna.Trial) -> tuple[str, dict]:
    family = trial.suggest_categorical(
        "family", ["ridge", "random_forest", "gradient_boosting"])
    if family == "ridge":
        return family, {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if family == "random_forest":
        return family, {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("rf_max_depth", 2, 12),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 50),
        }
    return family, {
        "n_estimators": trial.suggest_int("gb_n_estimators", 50, 400, step=50),
        "learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("gb_max_depth", 1, 4),
        "subsample": trial.suggest_float("gb_subsample", 0.5, 1.0),
    }


def tune_target(W: np.ndarray, target: np.ndarray, name: str, trials: int) -> dict:
    cv = KFold(DML_CV, shuffle=True, random_state=SEED)

    def cv_mse(model) -> float:
        return float(-cross_val_score(model, W, target, cv=cv,
                                      scoring="neg_mean_squared_error").mean())

    def objective(trial: optuna.Trial) -> float:
        family, params = _suggest(trial)
        return cv_mse(build_model(family, params))

    study = optuna.create_study(
        direction="minimize", study_name=name,
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    family = study.best_params["family"]
    prefix = {"random_forest": "rf_", "gradient_boosting": "gb_"}.get(family, "")
    params = {k.removeprefix(prefix): v for k, v in study.best_params.items()
              if k != "family" and (prefix == "" or k.startswith(prefix))}
    # baselines ~ econml 0.16's 'auto' candidates (['forest','linear'])
    base = {"lasso_cv": cv_mse(LassoCV(random_state=SEED)),
            "rf_default": cv_mse(RandomForestRegressor(random_state=SEED))}
    best = {"family": family, "params": params, "cv_mse": study.best_value,
            "baseline_cv_mse": base, "n_trials": trials}
    gain = (min(base.values()) - study.best_value) / min(base.values())
    print(f"\n[{name}] best: {family} {params}")
    print(f"[{name}] CV MSE {study.best_value:.6f}  vs 'auto'-candidates "
          f"lasso {base['lasso_cv']:.6f} / rf-default {base['rf_default']:.6f} "
          f"({gain:+.1%} vs best baseline)")
    return best


def main() -> int:
    p = argparse.ArgumentParser(description="Tune LinearDML nuisances (predictive CV only).")
    p.add_argument("--trials", type=int, default=CFG["tuning"]["trials"])
    args = p.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = load_frame()
    d = df.dropna(subset=["y_first_trip"])
    # W constructed EXACTLY as in estimate.run_pooled_dml
    W = pd.get_dummies(d[["region"]].assign(yr=d.trip_year.astype(int).astype(str)),
                       drop_first=True, dtype=float).to_numpy()
    print(f"-> tuning on n={len(d):,}, W has {W.shape[1]} dummy columns; "
          f"{args.trials} trials per nuisance, {DML_CV}-fold CV, seed {SEED}")

    result = {
        "model_y": tune_target(W, d.y_first_trip.to_numpy(float), "model_y", args.trials),
        "model_t": tune_target(W, d.hdi.to_numpy(float), "model_t", args.trials),
        "meta": {"n": len(d), "w_cols": int(W.shape[1]), "seed": SEED,
                 "outcome": "y_first_trip", "treatment": "hdi",
                 "cv_folds": DML_CV,
                 "objective": f"{DML_CV}-fold CV MSE (predictive only — never the causal estimate)"},
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH}")
    print("consume via: uv run python analysis/estimate.py --run --tuned-nuisances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
