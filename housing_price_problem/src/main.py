from pathlib import Path
from typing import Any
import pandas as pd
from src.config_loader import load_config
from src.logger import get_logger   
from src.split import split_data
from src.models import get_baseline_models, get_candidate_models
from src.train import train_model   
from src.evaluation import evaluate_models
from src.load_data import load_data_from_kaggle
from src.cross_validation import cross_validate_models

config = load_config()
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_model_collections(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, Any],
    config: dict,
    collection_name: str,
) -> dict[str, Any]:
    """
    Trains a collection of models and returns the trained models.
    it trains models one by one and logs the progress.
    """

    trained_models: dict[str, Any] = {}

    logger.info(f"Training {collection_name} models...")

    for model_name, model in models.items():
        try:
            logger.info(
                'Started training collection: %s, model: %s', collection_name, model_name
            )

            fitted_model = train_model(
                X_train=X_train,
                y_train=y_train,
                models={model_name: model},
                config=config
            )

            trained_models.update(fitted_model)
        except Exception as e:
            logger.exception(
                'Error occurred while training collection: %s, model: %s. Error: %s',
                collection_name,
                model_name,
                str(e)
            )

    logger.info(
        'Finished training collection: %s. Total models trained: %d. Failed models: %d',
        collection_name,
        len(trained_models),
        len(models) - len(trained_models)
    )

    return trained_models

def evaluate_model_collections(
    X_val : pd.DataFrame,
    y_val : pd.Series,
    trained_models: dict[str, Any],
    collection_name: str
) -> pd.DataFrame:
    '''
    Evaluates a collection of models and returns the evaluation results as a DataFrame. and labels the results with the collection name.
    '''
    if not trained_models:
        logger.warning("No trained models found for collection: %s. Skipping evaluation.", collection_name)
        return pd.DataFrame()  # Return an empty DataFrame if no models were trained

    results = evaluate_models(
        X_val=X_val,
        y_val=y_val,
        trained_models=trained_models,
    )

    results.insert(1, 'collection_name', collection_name)  # Insert collection name as the second column
    return results

def save_benchmark_results(
    results: pd.DataFrame,
    config: dict,
) -> Path:
    ''''
    Saves the benchmark results to a CSV file in the specified output directory.
    '''


    evaluation_config = config.get('evaluation', {})
    results_dir = Path(evaluation_config.get('results_dir', 'results'))
    results_filename = evaluation_config.get('results_filename', 'benchmark_results.csv')
    if results_dir.is_absolute():
        output_dir = results_dir
    else:
        output_dir = PROJECT_ROOT / results_dir

    output_path = output_dir / results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    results.to_csv(output_path, index=False)

    logger.info("Benchmark results saved to: %s", output_path)
    return output_path

def main()-> None:
    ''''
    Run the initial baseline and candidate model training and evaluation.
    '''

    logger.info('=' * 70)
    logger.info('Starting the housing price prediction benchmark process...')
    logger.info('=' * 70)

    # Load configurations
    config = load_config()
    logger.info("Configuration loaded successfully.")

    #----------------------------------------------
    # Load data from Kaggle
    #----------------------------------------------
    train_df, test_df = load_data_from_kaggle(
        dataset=config['data']['dataset'],
        data_path=config['data']['data_path']
    )
    target_column = config['data']['target_column']
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data.")

    logger.info("Data loaded successfully. Training data shape: %s, Test data shape: %s", train_df.shape, test_df.shape)
    logger.info("Target column: %s", target_column)

    #----------------------------------------------
    # Split the data into training and validation sets
    #----------------------------------------------
    X_train, X_val, y_train, y_val = split_data(
        data=train_df,
        validation_size=config['split']['validation_size'],
        random_state=config['split']['random_state']
    )

    logger.info("Data split into training and validation sets successfully.")
    logger.info("Training set shape: %s, Validation set shape: %s", X_train.shape, X_val.shape)
    logger.info('Target set shape: y_train %s | y_val %s', y_train.shape, y_val.shape)

    #----------------------------------------------
    # define models collecition
    #---------------------------------------------
    baseline_models = get_baseline_models(config)
    candidate_models = get_candidate_models(config)
    logger.info('Baseline model defined: %d',len(baseline_models))
    logger.info('Candidate model defined: %d', len(candidate_models))

    #-------------------------------------------------
    # Training baseline model
    #-------------------------------------------------
    trained_baseline_model = train_model_collections(
        X_train= X_train,
        y_train= y_train,
        models= baseline_models,
        config=config,
        collection_name= 'baseline'
    )

    #---------------------------
    # Evaluate baseline models
    #---------------------------
    baseline_results = evaluate_model_collections(
        X_val= X_val,
        y_val= y_val,
        trained_models= trained_baseline_model,
        collection_name='baseline'
    )
    if not baseline_results.empty:
        logger.info('Result of baseline models \n %s', baseline_results.to_string(index=False))

    #-------------------------------------
    #Training candidate model
    #-------------------------------------
    trained_candidate_model = train_model_collections(
        X_train=X_train,
        y_train=y_train,
        models= candidate_models,
        config=config,
        collection_name='candidate'
    )

    #----------------------------------
    #Evaluate candidate model
    #----------------------------------
    candidate_results = evaluate_model_collections(
        X_val=X_val,
        y_val=y_val,
        trained_models=trained_candidate_model,
        collection_name='candidate'
    )

    if not candidate_results.empty:
        logger.info('Candidate results \n %s', candidate_results.to_string(index=False))


    #----------------------
    # Combining the results
    #----------------------

    results_frame = [
        frame for frame in (baseline_results, candidate_results) if not frame.empty
    ]

    if not results_frame:
        raise RuntimeError('Benchmark ran without making any evaluation')

    all_results_frame = pd.concat(results_frame, ignore_index=True)

    all_results_frame = (all_results_frame.sort_values(
        by = ['status', 'rmsle'],
        ascending= [False, True],
        na_position= 'last'
    ).reset_index(drop=True)
    )

    #---------------------------------
    #Save benchmark results
    #---------------------------------
    save_benchmark_results(results= all_results_frame, config=config)

    #--------------------------------
    # Cross validation
    #-------------------------------
    top_model_names = [
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "XGBRegressor",
    ]
    cv_models = {
        name: candidate_models[name]
        for name in top_model_names
        if name in candidate_models
    }
    missing_models = set(top_model_names) - set(cv_models)

    if missing_models:
        logger.warning(
            "These models were not found in candidate_models: %s",
            sorted(missing_models),
        )

    if cv_models:
        cv_results = cross_validate_models(
            X_train=X_train,
            y_train= y_train,
            models= cv_models,
            config=config
        )

        logger.info(
            "Cross-validation results:\n%s",
            cv_results.to_string(index=False),
        )
        # Save CV results separately from the benchmark.
        results_dir = PROJECT_ROOT / "reports"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / "cross_validation_results.csv"

        cv_results.to_csv(
            results_path,
            index=False,
        )

        logger.info(
            "Cross-validation results saved to %s",
            results_path,
        )
    else:
        logger.error(
            "No candidate models available for cross-validation."
        )


    #-------------------------------------------
    #Report best evaluation
    #-------------------------------------------

    successful_evaluation = all_results_frame.loc[
        all_results_frame['status'] == 'success'
    ]
    if successful_evaluation.empty:
        raise RuntimeError('All the evaluation ends in failed')

    best_evaluation = successful_evaluation.iloc[0]

    logger.info('Best models evaluation : %s | group: %s | rmsle: %.6f', best_evaluation['model_name'], best_evaluation['collection_name'], best_evaluation['rmsle'])


if __name__ == '__main__':
    main()









        