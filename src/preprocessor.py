"""
src/preprocessor.py
===================
Full preprocessing pipeline — every step a senior ML engineer handles
for binary classification on tabular data.
 
Steps (in order):
  1.  Missing value imputation
  2.  Outlier detection & capping
  3.  Skewness correction (power transform)
  4.  Duplicate column removal
  5.  Collinearity removal
  6.  Scaling / normalisation
  7.  Class imbalance handling (SMOTE / oversampling)
  8.  Feature selection
 
Design principles:
  - Fit ONLY on training data, transform train+val+test
  - Sklearn Pipeline-compatible where possible
  - Every step logged and configurable via configs/config.yaml
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from src.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# Custom Transformers (sklearn-compatible)
# ────
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using IQR or Z-score method.
    Capping (Winsorization) is preferred over removal in most tabular ML tasks
    because it preserves sample size and doesn't lose information.
    """
    def __init__(self, method: str = 'iqr', threshold: float= 1.5):
        self.method = method
        self.thresold = threshold
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y = None):
        X = pd.DataFrame(X)
        for col in X.columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds_[col] = Q1 - self.thresold * IQR
                self.upper_bounds_[col] = Q3 + self.thresold * IQR
            elif self.method == 'zscore':
                mu, sigma = X[col].mean(), X[col].std()
                self.lower_bounds_[col] = mu - self.thresold * sigma
                self.upper_bounds_[col] = mu + self.thresold * sigma

        return self
    
    def transform(self, X,y = None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.lower_bounds_:
                X[col] = X[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col]
                )

        return X.values # because it is necessary to be an array for pipeline
    
class CollinearityRemover(BaseEstimator, TransformerMixin):
    """
    Removes one feature from each pair with |correlation| > threshold.
    Fitted on training set only.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.cols_to_drop_ = []
        self.feature_name_in_ = None

    def fit(self, X, y= None):
        X = pd.DataFrame(X)
        self.feature_name_in_ = X.columns.tolist()
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
        )

        self.cols_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        logger.info(
            f"CollinearityRemover: dropping {len(self.cols_to_drop_)} features "
            f"with |corr| > {self.threshold}: {self.cols_to_drop_}"
        )
        return self
    
    def transform(self,X,y= None):
        X = pd.DataFrame(X,columns=self.feature_name_in_)
        return X.drop(columns=self.cols_to_drop_, errors='ignore').values
    
    def get_feature_name_out(self):
        return [c for c in self.feature_name_in_ if c not in self.cols_to_drop_]

class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """
    Applies Yeo-Johnson power transform to features with |skew| > threshold.
    Yeo-Johnson works on both positive and negative values (unlike Box-Cox).
    """
    def __init__(self, threshold: float = 0.75, method: str = 'yeo-johnson'):
        self.threshold = threshold
        self.method = method
        self.skewed_cols_ = []
        self.pt_ = None
        self.n_features_ = None
    
    def fit(self, X,y= None):
        X = pd.DataFrame(X)
        self.n_features_ = X.shape[1]
        skewness = X.skew().abs()
        self.skewed_cols_ = skewness[skewness > self.threshold].index.tolist()
        logger.info(
            f"SkewnessCorrector: {len(self.skewed_cols_)} skewed features"
            f"(threeshold = {self.threshold}): {self.skewed_cols_}"
        )
        if self.skewed_cols_:
            self.pt_ = PowerTransformer(method=self.method, standardize=False)
            self.pt_.fit(X[self.skewed_cols_])
            return self
        
    def transform(self, X,y= None):
        X = pd.DataFrame(X).copy()
        if self.skewed_cols_ and self.pt_ is not None:
            X[self.skewed_cols_] = self.pt_.transform(X[self.skewed_cols_])
        return X.values
    

# ─────────────────────────────────────────────────────────────────
# Pipeline builders
# ─────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(config: dict)->Pipeline:

    """
    Assembles the sklearn preprocessing Pipeline from config.
 
    NOTE: Feature selection is done OUTSIDE this pipeline
    (after fitting) because it needs the target y.
 
    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pp_cfg = config['preprocessing']

    # --- Imputer ---
    strategy = pp_cfg.get('missing_strategy','median')
    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy)

    # --- Outlier capper ---
    outlier_capper = OutlierCapper(
        method=pp_cfg.get('outlier_method', 'iqr'),
        threshold= pp_cfg.get('outlier_threshold', 1.5),
    )

    # --- Skewness corrector ---
    skew_corrector = SkewnessCorrector(
        threshold=pp_cfg.get('skewness_threshold', 0.75),
        method=pp_cfg.get('skewness_transform', 'yeo-johnson'),
    )
    
    # --- Collinearity remover ---
    collinearity_remover = CollinearityRemover(
        threshold=pp_cfg.get('collinearity_threshold', 0.95)
    )

    # --- Scaler ---
    scaling = pp_cfg.get('scaling_method', 'standard')
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
    }
    scaler = scalers.get(scaling, StandardScaler())

    steps = [
        ('imputer', imputer),
        ('outlier_capper', outlier_capper),
        ('skewness_corrector', skew_corrector),
        ('collinearity_remover', collinearity_remover),
        ('scaler',scaler)
    ]

    pipeline = Pipeline(steps)
    logger.info(f"Preprocessing pipeline built with {len(steps)} steps.")
    return pipeline

def fit_transform_pipeline(
        pipeline: Pipeline,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
)-> tuple:
    """
    Fit on train, transform all splits.
    This is the ONLY correct way to prevent data leakage.
 
    Returns
    -------
    X_train_t, X_val_t, X_test_t (all np.ndarray)
    """
    logger.info("Fitting preprocessing pipeline on TRAINING data only...")

    X_train_t = pipeline.fit_transform(X_train)
    logger.info("Transforming validation and test sets...")
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    logger.info(
        f"Post-preprocessing shapes → Train: {X_train_t.shape} | "
        f"Val: {X_val_t.shape} | Test: {X_test_t.shape}"
    )
    return X_train_t, X_val_t, X_test_t
    
# ─────────────────────────────────────────────────────────────────
# Feature Selection (fit on train only)
# ─────────────────────────────────────────────────────────────────

def select_features(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        method: str='rfe',
        feature_names: list= None,
)-> tuple:
    """
    Feature selection fitted on training data only.
 
    Methods:
      - rfe          : Recursive Feature Elimination with LogisticRegression
      - selectkbest  : Univariate statistical test (ANOVA F-value)
      - lasso        : L1-regularised Logistic Regression coefficients
      - none         : No selection (return as-is)
 
    Returns
    -------
    X_train_sel, X_val_sel, X_test_sel, selected_feature_names, selector
    """

    logger.info(f'Features selection method: {method}')

    if method == 'rfe':
        estimator = LogisticRegression(
            max_iter=10000, random_state=42, solver='liblinear'
        )
        selector = RFE(estimator=estimator, n_features_to_select=0.7, step=1)

    elif method == 'selectkbest':
        k = max(1, int(X_train.shape[1] * 0.7))
        selector = SelectKBest(score_func=f_classif, k = k)

    elif method == 'lasso':
        lasso = LogisticRegression(
            penalty='l1', C=0.01, solver='liblinear', 
            max_iter=10000, random_state=42
        )
        selector = SelectFromModel(lasso, threshold='mean')

    else:
        logger.info('No feature selection applied.')
        selected_names = feature_names or list(range(X_train.shape[1]))
        return X_train, X_val,X_test, selected_names, None
    
    selector.fit(X_train,y_train)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    if feature_names is not None:
        mask = selector.get_support()
        selected_names = [f for f,m in zip(feature_names, mask) if m]
    else:
        selected_names = list(range(X_train_sel.shape[1]))

    logger.info(
        f"Feature selection: {X_train.shape[1]} → {X_train_sel.shape[1]} features selected."
    )
    logger.info(f"Selected features: {selected_names}")
 
    return X_train_sel, X_val_sel, X_test_sel, selected_names, selector


# ─────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────
 
def save_preprocessor(pipeline: Pipeline, path: str = 'models/preprocessor.joblib')->None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Preprocessor saved to: {path}")

def load_preprocessor(path: str = 'models/preprocessor.joblib')-> Pipeline:
    pp = joblib.load(path)
    logger.info(f'Preprocessor loaded from: {path}')
    return pp
