from src.logger import get_logger
from src.config_loader import load_config


from sklearn.model_selection import train_test_split
import pandas as pd


logger = get_logger(__name__)
config = load_config()

def split_data(data: pd.DataFrame, validation_size: float = config['split']['validation_size'], random_state: int = config['split']['random_state']) -> tuple:
    """
    Splits the data into training and validation sets.

    Parameters:
    - data: The DataFrame to split.
    - validation_size: The proportion of the dataset to include in the validation split.
    - random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
    - A tuple containing the training and validation DataFrames.
    """
    
    df = data.copy()
    X = df.drop(columns=[config['data']['target_column']])
    y = df[config['data']['target_column']]

    if config['split']['strategy'] == 'random':
        logger.info("Using random split strategy.")
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=validation_size, random_state=random_state, shuffle=True)
    else:
        raise NotImplementedError(f"Split strategy '{config['split']['strategy']}' is not implemented.")
    
    logger.info(f"Data split into training and validation sets with validation size {validation_size}")
    
    return X_train, X_val, y_train, y_val