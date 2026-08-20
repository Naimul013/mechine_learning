from src.logger import get_logger
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from pathlib import Path
import zipfile
import logging

logger = get_logger(__name__)
logger.info("load_data logger is working")

def load_data_from_kaggle(dataset: str= 'house-prices-advanced-regression-techniques', data_path : str='data/raw') -> tuple[pd.DataFrame, pd.DataFrame]:

    path = Path(data_path)
    path.mkdir(parents = True, exist_ok = True)

    train_path = path / 'train.csv'
    test_path = path / 'test.csv'

    if not train_path.exists() or not test_path.exists():
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(dataset, path = path, quiet = False)

    zip_path = path / f'{dataset}.zip'
    if zip_path.exists():
        logger.info(f'Extracting train and test data from {zip_path}')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        zip_path.unlink()  # Remove the zip file after extraction

    logger.info(f'Loading data from {path}')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Check if the dataframes are empty
    logger.info(f'Train DataFrame shape: {train_df.shape}')
    logger.info(f'Test DataFrame shape: {test_df.shape}')

    return train_df, test_df
        

    


