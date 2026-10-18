from matplotlib.pylab import Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config_loader import load_config
from src.logger import get_logger
from src.preprocessor import build_preprocessor

config = load_config()
logger = get_logger(__name__)

models_need_scaling = {
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'KNeighborsRegressor',
}

def model_pipeline(model_name: str, models: Any, config: dict) -> Pipeline:
    """
    Creates a pipeline that combines the preprocessor and the model.
    """
    preprocessor = build_preprocessor(config)
    steps=[
            ('preprocessor', preprocessor),
        ]
    
    if model_name in models_need_scaling:
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', models))
    
    return Pipeline(steps=steps)


def train_model(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        models : Any,
        config : dict,
) -> dict[str, Pipeline]:
    """
    Creates a dictionary of trained models for regression tasks.
    """
    trained_models : dict[str, Pipeline] = {}
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        pipeline = model_pipeline(
            model_name=model_name,
            models=model,
            config=config
        )
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        logger.info(f"{model_name} trained successfully.")

    return trained_models

    
    