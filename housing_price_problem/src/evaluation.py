from collections.abc import Mapping
import pandas as pd
import numpy as np
from traitlets import Any
from src.config_loader import load_config
from src.logger import get_logger

from sklearn.metrics import root_mean_squared_log_error
from sklearn.pipeline import Pipeline

config = load_config()
logger = get_logger(__name__)

def validate_evaluation_data(
        X_val : pd.DataFrame,
        y_val : pd.Series,

) -> np.ndarray:
    ''''
    Validates the evaluation data to ensure it meets the expected criteria.
    '''

    if not isinstance(X_val, pd.DataFrame):
        raise ValueError("X_val should be a pandas DataFrame.")

    if X_val.empty:
        raise ValueError("X_val is empty.")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val should have the same number of samples.")

    y_true = np.asarray(y_val, dtype=np.float64).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_val is empty.")
    if not np.isinfinite(y_true).all() and not np.isnan(y_true).all():
        raise ValueError("y_val contains NaN or infinite values.")
    if y_true.any() < 0:
        raise ValueError("y_val contains negative values, which is unexpected for housing prices.")

    return y_true

RESULT_COLUMNS = [
    "model_name",
    "rmsle",
    "status",
    "negative_predictions",
    "prediction_min",
    "prediction_max",
    "error",
]

def evaluate_models(
        trained_models : Mapping[str, Pipeline],
        X_val : pd.DataFrame,
        y_val : pd.Series,
) -> pd.DataFrame:
    '''
    Evaluates the trained models on the validation data and returns a DataFrame with the evaluation results.
    '''

    if not trained_models:
        raise ValueError("No trained models provided for evaluation.")
    y_true = validate_evaluation_data(X_val, y_val)

    logger.info("Evaluating models...")
    logger.info('Number of models to evaluate: %d', len(trained_models))
    logger.info('Number of validation samples: %d', len(y_true))

    results: list[dict[str, Any]] = []
    for model_name, model in trained_models.items():
        logger.info('Evaluating model: %s', model_name)

        result: dict[str, Any] = {
            'model_name': model_name,
            'rmsle': np.nan,
            'status':'failed',
            'negative_predictions': np.nan,
            'prediction_min': np.nan,
            'prediction_max': np.nan,
            'error': None,
        }

        try:
            # start evaluation
            predictions = np.asarray(
                model.predict(X_val),
                dtype=np.float64,
            ).reshape(-1)
            if predictions.size != y_true.size:
                raise ValueError(f"Predictions size {predictions.size} does not match y_true size {y_true.size}.")

            if not np.isfinite(predictions).all():
                raise ValueError("Predictions contain NaN or infinite values.")

            negative_predictions = int((predictions < 0).sum())

            # house prices should not be negative, but if they are, we will set them to zero for RMSLE calculation
            valid_predictions = np.maximum(predictions, 0.0)
            rmsle = float(
                root_mean_squared_log_error(
                    y_true,
                    valid_predictions
                )
            )
            result.update({
                'rmsle': rmsle,
                'status':'success',
                'negative_predictions': negative_predictions,
                'prediction_min': float(np.min(valid_predictions)),
                'prediction_max': float(np.max(valid_predictions)),
            })

            logger.info('Model evaluation completed successfully: %s | RMSLE: %.6f'
                        'Negative predictions: %d', model_name, rmsle, negative_predictions)

        except Exception as e:
            result['error'] = f'{type(e).__name__}: {str(e)}'
            logger.exception('Error evaluating model %s: %s', model_name, result['error'])

        results.append(result)

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    # sort the results by RMSLE, with successful evaluations first
    results_df = results_df.sort_values(by=['status', 'rmsle'], ascending=[False, True], na_position='last').reset_index(drop=True)

    successful_count = (results_df['status'] == 'success').sum()
    negative_count = len(results_df) - successful_count
    logger.info('Evaluation completed. Successful evaluations: %d, Failed evaluations: %d', successful_count, negative_count)

    if successful_count:
        best_model = results_df.loc[results_df['status'] == 'success'].iloc[0]
        logger.info('Best model: %s | RMSLE: %.6f | Negative predictions: %d', best_model['model_name'], best_model['rmsle'], best_model['negative_predictions'])

    else:
        logger.warning('No successful model evaluations. Please check the errors in the evaluation results.')

    return results_df


    


    
