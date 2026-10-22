"""Evaluate the tuned model on the held-out validation partition."""

from pathlib import Path

import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_log_error

from src.config_loader import load_config
from src.logger import get_logger
from src.load_data import load_data_from_kaggle
from src.models import get_candidate_models
from src.split import split_data
from src.train import model_pipeline


config = load_config()
logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_path(path: str)-> Path:
    path = Path(path).expanduser()

    return path if path.is_absolute() else PROJECT_ROOT/path


def calculate_rmsle(y_true, y_predict)-> float:
    y_true = np.asarray(y_true)
    y_predict = np.asarray(y_predict)
    if np.any(y_true<0):
        raise ValueError("RMSLE requires nonnegative target values.")
    y_predict = np.clip(y_predict, 0, None)
    return float(root_mean_squared_log_error(y_true=y_true, y_pred=y_predict))

def main():
    config = load_config()
    tuning_config = config["tuning"]

    # Load the existing study. This does NOT run additional trials.
    storage_path = resolve_path(tuning_config['storage_path'])
    study = optuna.load_study(
        study_name=tuning_config['study_name'],
        storage=f'sqlite:///{storage_path.as_posix()}'
    )

    model_name = tuning_config['model_name']
    candidate_models = get_candidate_models(config)

    if model_name not in candidate_models:
        raise KeyError('%s model name not found', model_name)

    # Recreate the same train/validation split used by the project.
    train_df,_ = load_data_from_kaggle(
        dataset=config["data"]["dataset"],
        data_path=config["data"]["data_path"],
    )

    X_train, X_val , y_train, y_val = split_data(
        data=train_df, validation_size= config['split']['validation_size'],
        random_state=config['split']['random_state']
    )

    # Baseline model: default candidate-model parameters.
    baseline_pipeline = model_pipeline(
        model_name=model_name,
        models= clone(candidate_models[model_name]),
        config=config,
    )

    tuned_estimator = clone(candidate_models[model_name]).set_params(**study.best_params)
    tuned_pipeline = model_pipeline(
        model_name=model_name,
        models= tuned_estimator,
        config=config,
    )

    logger.info("Fitting baseline model...")
    baseline_pipeline.fit(X_train, y_train)

    logger.info("Fitting tuned model...")
    tuned_pipeline.fit(X_train, y_train)

    baseline_predictions = baseline_pipeline.predict(X_val)
    tuned_predictions = tuned_pipeline.predict(X_val)

    baseline_rmsle = calculate_rmsle(y_true=y_val,y_predict = baseline_predictions)
    tuned_rmsle = calculate_rmsle(y_true= y_val, y_predict= tuned_predictions)

    improvement = baseline_rmsle - tuned_rmsle
    percentage_improvement = (
        100 * improvement / baseline_rmsle
        if baseline_rmsle > 0
        else 0.0
    )

    results = pd.DataFrame(
        [
            {
                "model": f"{model_name}_baseline",
                "validation_rmsle": baseline_rmsle,
            },
            {
                "model": f"{model_name}_optuna_tuned",
                "validation_rmsle": tuned_rmsle,
            },
        ]
    )

    output_path = resolve_path(
        config["evaluation"]["tuned_result_path"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    logger.info("Baseline validation RMSLE: %.6f", baseline_rmsle)
    logger.info("Tuned validation RMSLE: %.6f", tuned_rmsle)
    logger.info(
        "Relative improvement: %.2f%%",
        percentage_improvement,
    )
    logger.info("Validation comparison saved to %s", output_path)


if __name__ == "__main__":
    main()
    

