"""
src/data_loader.py
==================
Handles data ingestion, validation, and initial profiling.
In production this would connect to databases, S3, feature stores, etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
 
def load_raw_data(save_to_disk: bool = True, raw_path: str = 'data/raw/') -> pd.DataFrame:
    """
    Load the breast cancer dataset and optionally persist to disk.
    In production: replace this with your DB / S3 / feature-store connector.
 
    Returns
    -------
    pd.DataFrame  with features + 'target' column
    """

    logger.info('Loading breast cancer dataset from sklearn......')
    bc = load_breast_cancer(as_frame=True)

    df = bc.frame.copy()
    df['target'] = df['target']

    logger.info(f'Dataset shape: {df.shape}')
    logger.info(f"Target distribution:\n{df['target'].value_counts().to_string()}")

    if save_to_disk:
        Path(raw_path).mkdir(parents=True, exist_ok=True)
        out= Path(raw_path)
        df.to_csv(out, index = False)
        logger.info(f'Raw data saved to: {out}')

    return df

# ─────────────────────────────────────────────
# 2. Validate schema & types
# ─────────────────────────────────────────────
 

def validate_data(df: pd.DataFrame, target_col: str = 'target') -> None:
    """
    Run basic schema / data-quality assertions.
    In production: use Great Expectations or Pandera.
    """

    logger.info('Running data validation check....')

    assert df.shape[0] > 0, 'Dataframe is empty.' # if it is false stop program with the error dataframe is empty
    assert target_col in df.columns, f'Target column missing '
    assert df[target_col].nunique() == 2, 'Target must be binary'
    assert df.duplicated().sum() == 0, 'Dplicate found'

    numeric_col = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_col = [c for c in df.columns if c not in numeric_col or c != target_col]

    if non_numeric_col:
        logger.warning(f'Non numeric columns found: {non_numeric_col} - will need to encoding')
    else:
        logger.info('All features are numeric')

    missing_pct = df.isnull().mean().mul(100).round(2)
    high_missing = missing_pct[missing_pct > 5]
    if not high_missing.empty:
        logger.warning(f"Columns with >5% missing:\n{high_missing.to_string()}")
    else:
        logger.info("No columns with >5% missing values. ✅")

    logger.info('Data validation complete.')


# ─────────────────────────────────────────────
# 3. Split — DONE ONCE, NEVER TOUCHED AGAIN
# ──────────────────────────

def split_data(
        df: pd.DataFrame,
        target_col: str = 'target',
        test_size: float = 0.20,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
)-> tuple:
    """
    Chronological split rule:
      1. Split off holdout TEST set first — never touch until final evaluation.
      2. Split remaining into TRAIN + VALIDATION.
 
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
     
    logger.info('Splitting the data.....')

    X = df.drop(columns=[target_col])
    y = df[target_col]

    strat = y if stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
         X,y, test_size=test_size,random_state= random_state,stratify=strat
    )

    val_frac = val_size / (1-test_size)
    strat2 = y_trainval if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size= val_frac,random_state=random_state,
        stratify=strat2
    )

    logger.info(f"Train size : {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"Val size   : {X_val.shape[0]}   ({X_val.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"Test size  : {X_test.shape[0]}  ({X_test.shape[0]/len(df)*100:.1f}%)")
 
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_split.value_counts(normalize=True).round(3)
        logger.info(f"{name} target distribution: {dist.to_dict()}")
 
    return X_train, X_val, X_test, y_train, y_val, y_test




