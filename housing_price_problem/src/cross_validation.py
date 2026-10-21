from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.logger import get_logger
from src.config_loader import load_config
from src.train import model_pipeline

logger = get_logger(__name__)
config = load_config()

CV_RESULT_COLUMNS = [
    'model_name',
    'cv_rmsle_mean',
    'cv_rmsle_std',
    'cv_rmsle_min',
    'cv_rmsle_max',
    'fit_time_mean',
    'score_time_mean',
    'status',
    'error',
]

def _negative_rmsle_scorer(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Calculate negative RMSLE for scikit-learn's scoring convention.

    Predictions below zero are clipped to zero because RMSLE requires
    non-negative values. The original estimator is not modified.
    """
    y_true = np.asarray(y, dtype=np.float64,).reshape(-1)
    predictions = np.asarray(
        estimator.predict(X),
        dtype= np.float64
    ).reshape(-1)
    if predictions.shape != y_true.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{predictions.shape} != {y_true.shape}"
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            'Target values contain nan or infinite value'
        )
    if not np.isfinite(predictions).all():
        raise ValueError(
            'Predictions values contain nan or infinite value'
        )

    if (y_true<0).any():
        raise ValueError(
            'rmsle scoring needs non negative value'
        )
    predictions = np.maximum(predictions, 0.0)
    rmsle = root_mean_squared_log_error(
        y_true, predictions
    )

    return -float(rmsle)

def _validate_cv_input(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    models: Mapping[str, Any],
) -> np.ndarray:

    """
    Validate cross-validation inputs and return the original target
    as a one-dimensional NumPy array.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError(
            'X values must be a pd dataframe'
        )

    if not isinstance(y_train,(pd.Series,pd.DataFrame, np.ndarray,list)):
        raise TypeError(
            'ytrain must be one of the above sequence'
        )

    if X_train.empty:
        raise ValueError(
            'Train data is empty'
        )

    y_values = np.asarray(
        y_train, dtype= np.float64,
    ).reshape(-1)

    if len(X_train) != len(y_values):
        raise ValueError(
            'Xtrain and ytrain must contain the same number of observation'
        )

    if not models:
        raise ValueError(
            'There should not be 0 models'
        )

    return y_values

def _build_stratified_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    random_state: int | None, 
    shuffle: bool,
)-> list[tuple[np.ndarray,np.ndarray]]:
    """
    Build regression CV folds stratified using target-price bins.

    The bins are used only to determine fold membership. They are
    never supplied to the regressor as training targets.
    """
    number_of_bins = 10

    # Rank before qcut so repeated house prices at quantile
    # boundaries do not cause duplicate quantile edges.

    ranked_target = pd.Series(y_train).rank(
        method='first',
    )

    y_bins = pd.qcut(
        ranked_target, q= number_of_bins,labels= False,
    )
    y_bins = np.asarray(y_bins, dtype= np.float64)
    bin_counts = pd.Series(y_bins).value_counts()

    if len(bin_counts) < 2:
        raise ValueError(
            "Target binning produced fewer than two bins."
        )

    if int(bin_counts.min()) < n_splits:
        raise ValueError(
            "At least one target bin has fewer observations "
            "than n_splits. Reduce the number of folds or "
            "target bins."
        )

    logger.info(
        "Target-binned stratification enabled: %d bins.",
        len(bin_counts),
    )

    logger.debug(
        "Target-bin observation counts: %s",
        bin_counts.sort_index().to_dict(),
    )

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state= random_state,
    )

    # Generate indices from the target bins, while retaining
    # the original continuous target for model training.
    folds = list(
        splitter.split(X_train, y_bins,)
    )

    logger.info(
        "Generated %d stratified folds successfully.",
        len(folds),
    )

    return folds

def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Mapping[str,Any],
    config: dict,
)-> pd.DataFrame:

    """
    Cross-validate complete preprocessing-and-model pipelines.

    Parameters
    ----------
    X_train:
        Training features only. Do not pass X_val or test features.
    y_train:
        Original continuous SalePrice values.
    models:
        Mapping of model names to unfitted estimators.
    config:
        Project configuration loaded from YAML.

    Returns
    -------
    pd.DataFrame
        One summary row per model, including mean and standard
        deviation of fold RMSLE scores.
    """

    cv_config = config.get(
        'cross_validation',{}
    )
    n_splits = cv_config.get(
        'n_splits', 5
    )
    shuffle = cv_config.get(
        'shuffle', True
    )
    random_state = cv_config.get(
        'random_state', 42
    )
    n_jobs = cv_config.get(
        'n_jobs', 1
    )

    if not isinstance(shuffle, bool):
        raise TypeError(
            "cross_validation.shuffle must be a boolean."
        )

    if (
        not isinstance(n_jobs, int)
        or isinstance(n_jobs, bool)
        or n_jobs == 0
    ):
        raise ValueError(
            "cross_validation.n_jobs must be a non-zero integer."
        )

    if not shuffle and random_state is not None:
        logger.warning(
            "shuffle=False; random_state will not affect "
            "the fold assignments."
        )

    y_values = _validate_cv_input(
        X_train, y_train, n_splits=n_splits, models=models
    )
    cv_folds = _build_stratified_folds(
        X_train=X_train, y_train=y_train, n_splits=n_splits, random_state= random_state, shuffle=shuffle
    )
    logger.info(
        "Starting stratified cross-validation."
    )
    logger.info(
        "Number of models: %d",
        len(models),
    )
    logger.info(
        "Number of folds: %d",
        n_splits,
    )
    logger.info(
        "Training observations: %d",
        len(X_train),
    )

    results :list[dict[str,Any]] = []

    for model_name, model in models.items():
        logger.info(
            "Cross-validating model: %s",
            model_name,
        )

        result: dict[str, Any]={
            'model_name': model_name,
            "cv_rmsle_mean": np.nan,
            "cv_rmsle_std": np.nan,
            "cv_rmsle_min": np.nan,
            "cv_rmsle_max": np.nan,
            "fit_time_mean": np.nan,
            "score_time_mean": np.nan,
            "status": "failed",
            "error": None,
        }
        try:
            pipeline = model_pipeline(
                model_name=model_name,
                models= model,
                config=config
            )
            scores = cross_validate(
                estimator=pipeline,
                X=X_train,
                y= y_train, # original target value 
                cv= cv_folds,
                scoring= _negative_rmsle_scorer,
                n_jobs= n_jobs,
                return_train_score=False,
                error_score= 'raise',   
            )

            # sklearn scorers use higher-is-better.
            # Convert negative RMSLE scores to positive RMSLE.

            fold_rmsle = np.asarray(
                scores['test_score'],
                dtype=np.float64
            )

            if not np.isfinite(fold_rmsle).all():
                    raise ValueError(
                        "One or more folds returned a non-finite score."
                    )

            if len(fold_rmsle) != n_splits:
                raise ValueError(
                    f"Expected {n_splits} fold scores, "
                    f"received {len(fold_rmsle)}."
                )

            result.update(
                    {
                        "cv_rmsle_mean": float(
                            np.mean(fold_rmsle)
                        ),
                        "cv_rmsle_std": float(
                            np.std(fold_rmsle, ddof=1)
                        ),
                        "cv_rmsle_min": float(
                            np.min(fold_rmsle)
                        ),
                        "cv_rmsle_max": float(
                            np.max(fold_rmsle)
                        ),
                        "fit_time_mean": float(
                            np.mean(scores["fit_time"])
                        ),
                        "score_time_mean": float(
                            np.mean(scores["score_time"])
                        ),
                        "status": "success",
                    }
                )
            logger.info(
                "CV successful: %s | Mean RMSLE: %.6f | "
                "Std: %.6f | Min: %.6f | Max: %.6f",
                model_name,
                result["cv_rmsle_mean"],
                result["cv_rmsle_std"],
                result["cv_rmsle_min"],
                result["cv_rmsle_max"],
            )
        except Exception as exc:
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Cross-validation failed for model: %s",
                model_name,
            )

        results.append(result)

    results_df = pd.DataFrame(
        results,
        columns=CV_RESULT_COLUMNS,
    )

    results_df = (
        results_df.sort_values(
            by=["status", "cv_rmsle_mean"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    successful_count = int(
        (results_df["status"] == "success").sum()
    )

    failed_count = len(results_df) - successful_count

    logger.info(
        "Cross-validation completed. "
        "Successful: %d | Failed: %d",
        successful_count,
        failed_count,
    )

    if successful_count:
        best_model = results_df.loc[
            results_df["status"] == "success"
        ].iloc[0]

        logger.info(
            "Best CV model: %s | Mean RMSLE: %.6f | "
            "Std: %.6f",
            best_model["model_name"],
            best_model["cv_rmsle_mean"],
            best_model["cv_rmsle_std"],
        )
    else:
        logger.warning(
            "No models completed cross-validation successfully."
        )

    return results_df







    

    
    


    