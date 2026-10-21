from __future__ import annotations

import optuna
import pandas as pd
import numpy as np

import json
from pathlib import Path
from typing import Any
import joblib

from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.config_loader import load_config
from src.load_data import load_data_from_kaggle
from src.logger import get_logger
from src.cross_validation import (
    _build_stratified_folds, _negative_rmsle_scorer
)
from src.split import split_data
from src.models import get_candidate_models
from src.train import model_pipeline

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTEGER_PARAMETERS = {
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
}

def resolve_project_path(path: str)-> Path:
    configured_path = Path(path).expanduser()
    if configured_path.is_absolute():
        return configured_path

    return PROJECT_ROOT/configured_path

def suggest_parameters(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
)-> dict[str, Any]:
    """Generate one parameter combination from the configured search space."""

    parameters : dict[str, Any] = {}

    for name, bounds in search_space.items():
        low = bounds['low']
        high = bounds['high']
        if name in INTEGER_PARAMETERS:
            parameters[name] = trial.suggest_int(
                name=name,
                low=int(low),
                high=int(high),
            )
        else:
            parameters[name] = trial.suggest_float(
                name=name,
                low= float(low),
                high=float(high),
                log= bool(bounds.get('log', False)),
            )

    return parameters

def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    models: dict[str, Any],
    config: dict[str, Any],
)-> tuple[optuna.Study,Any]:

    """Tune one configured model using cross-validated positive RMSLE."""
    tuning_config = config['tuning']
    cv_config = config['cross_validation']
    model_name = tuning_config['model_name']

    if model_name not in models:
        raise KeyError(
            f"Model {model_name!r} was not found in candidate_models. "
            f"Available models: {sorted(models)}"
        )

    n_trials = tuning_config['n_trials']
    if n_trials < 1:
        raise ValueError(
            'tuning.n_trials must be at least 1.'
        )

    search_space = tuning_config['search_space']
    if not search_space:
        raise ValueError("tuning.search_space must not be empty.")

    # Use the same target-stratified fold strategy as the existing CV module.
    cv_folds = _build_stratified_folds(
        X_train= X_train,
        y_train= y_train,
        n_splits=int(cv_config['n_splits']),
        shuffle=bool(cv_config['shuffle']),
        random_state= int(cv_config['random_state'])
    )

    base_model = models[model_name]

    def objective(trial: optuna.Trial)-> float:
        parameters = suggest_parameters(trial, search_space)
        # Clone the original estimator so trials cannot modify shared state.
        clone_model = clone(base_model).set_params(**parameters)

        # The preprocessor is inside the pipeline, so each fold fits its
        # own imputers, encoders, and other learned transformations.
        pipeline = model_pipeline(
            model_name=model_name,
            models= clone_model,
            config=config
        )

        folds_result = cross_validate(
            estimator=pipeline,
            X= X_train,
            y=y_train,
            cv=cv_folds,
            scoring= _negative_rmsle_scorer,
            n_jobs= int(cv_config.get('n_jobs', 1)),
            return_train_score= False,
            error_score= 'raise'
        )

        # scikit-learn negates loss metrics; Optuna minimizes positive RMSLE.
        fold_rmsle = -folds_result["test_score"]
        mean_rmsle = float(fold_rmsle.mean())

        trial.set_user_attr("fold_rmsle_std", float(fold_rmsle.std(ddof=1)))
        trial.set_user_attr("fold_rmsle_min", float(fold_rmsle.min()))
        trial.set_user_attr("fold_rmsle_max", float(fold_rmsle.max()))

        logger.info(
            "Trial %d | mean RMSLE=%.6f | std=%.6f | parameters=%s",
            trial.number,
            mean_rmsle,
            float(fold_rmsle.std(ddof=1)),
            parameters,
        )

        return mean_rmsle

    storage_path = resolve_project_path(tuning_config['storage_path'])
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed= int(tuning_config['random_state']) # TPESampler is Optuna's smart AI optimizer for picking hyperparameters,
        )

    study = optuna.create_study(
        study_name= tuning_config['study_name'],
        storage= f'sqlite:///{storage_path.as_posix()}', # file system use forward slash , windows use backward slash
        sampler=sampler,
        direction= 'minimize', #Instructs Optuna's algorithm to search for hyperparameters that produce the lowest possible output value from objective(trial).
        load_if_exists= True,

    )
    logger.info(
        "Starting Optuna study %s with up to %d additional trials.",
        tuning_config["study_name"],
        n_trials,
    )

    study.optimize( #triggers Optuna's hyperparameter search loop
        objective,
        n_trials=n_trials,
        timeout=tuning_config.get("timeout"),
        gc_after_trial=True,
        show_progress_bar=True,
    )

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise RuntimeError(
            "Optuna finished without a successful trial. "
            "Inspect the logs and failed trial details."
        )

    logger.info(
        "Best CV RMSLE: %.6f | Best parameters: %s",
        study.best_value,
        study.best_params,
    )

    return study, base_model

def save_tuning_artifacts(
    study: optuna.Study,
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> None:
    """Save trial history, best parameters, and a fitted best pipeline."""
    tuning_config = config["tuning"]
    model_name = tuning_config["model_name"]

    best_params_path = resolve_project_path(
        tuning_config["best_params_filename"]
    )
    trials_path = resolve_project_path(tuning_config["trials_filename"])
    model_path = resolve_project_path(
        tuning_config["model_artifact_path"]
    )

    for path in (best_params_path, trials_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Include both the winning parameters and experiment metadata.
    best_result = {
        "study_name": study.study_name,
        "model_name": model_name,
        "best_cv_rmsle": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": study.best_params,
        "n_completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ),
        "n_failed_trials": sum(
            trial.state == optuna.trial.TrialState.FAIL
            for trial in study.trials
        ),
    }

    with best_params_path.open("w", encoding="utf-8") as file:
        json.dump(best_result, file, indent=4)

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "state",
            "datetime_start",
            "datetime_complete",
            "duration",
            "user_attrs",
        )
    )
    trials_df.to_csv(trials_path, index=False)

    # Fit the winning pipeline on the training partition only.
    best_estimator = clone(base_model).set_params(**study.best_params)
    best_pipeline = model_pipeline(
        model_name=model_name,
        models=best_estimator,
        config=config,
    )
    best_pipeline.fit(X_train, y_train)

    joblib.dump(best_pipeline, model_path)

    logger.info("Best parameters saved to %s", best_params_path)
    logger.info("Trial history saved to %s", trials_path)
    logger.info("Fitted best pipeline saved to %s", model_path)


def main() -> None:
    """Run the configured Optuna tuning workflow."""
    config = load_config()

    train_df, _ = load_data_from_kaggle(
        dataset=config["data"]["dataset"],
        data_path=config["data"]["data_path"],
    )

    X_train, _, y_train, _ = split_data(
        data=train_df,
        validation_size=config['split']['validation_size'],
        random_state=config['split']['random_state']
    )

    candidate_models = get_candidate_models(config)

    study, base_model = tune_model(
        X_train=X_train,
        y_train=y_train,
        models=candidate_models,
        config=config,
    )

    save_tuning_artifacts(
        study=study,
        base_model=base_model,
        X_train=X_train,
        y_train=y_train,
        config=config,
    )

    logger.info(
        "Tuning completed. Best mean CV RMSLE: %.6f",
        study.best_value,
    )


if __name__ == "__main__":
    main()



        

