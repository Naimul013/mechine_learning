"""
src/models.py
=============
Model zoo, training, cross-validation, and hyperparameter tuning.
Every model a senior ML engineer benchmarks for binary classification.
"""

import numpy as np
import pandas as pd
import joblib
import optuna
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_validate,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, matthews_corrcoef,
    brier_score_loss
)

from src.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1.  Model Zoo
#

def get_baseline_models()-> Dict[str, Any]:
    """
    Baselines every serious ML project must beat.
    If your model can't beat these, something is wrong.
    """

    return {
        'DummyClassifier (most_frequent)' : DummyClassifier(strategy='most_frequent'),
        'DummyClassifier (stratified)' : DummyClassifier(strategy='stratified', random_state=42),
        'LogisticRegression (baseline)' : LogisticRegression(
            max_iter=10000, random_state=42, solver='lbfgs'
        ),
        'GaussianNB (baseline)' : GaussianNB(),
        'LDA' : LinearDiscriminantAnalysis(),
    }

def get_candidate_models(random_state: int= 42) -> Dict[str, Any]:
    """
    Full candidate model zoo for binary classification.
    Default hyperparameters — we tune the winner later.
    """

    return {
        #Liner models
        'LogisticRegression': LogisticRegression(
            max_iter=10000, random_state=random_state, solver = 'lbfgs', C=1.0
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=10000, random_state=random_state, solver='liblinear',
            penalty='l1', C=1.0
        ),
        'SDGClassifier': SGDClassifier(
            random_state=random_state, max_iter=10000, loss='log_loss'
        ),
        'LDA' : LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),

        # Distance-based
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=11)': KNeighborsClassifier(n_neighbors=11),

        # SVM
        'SVC (RBF)': SVC(probability=True, random_state=random_state, kernel='rbf'),
        'SVC (Linear)': SVC(probability=True, random_state=random_state, kernel='linear'),

        # Tree-based
        'DecisionTree': DecisionTreeClassifier(random_state=random_state, max_depth=5),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, random_state=random_state, learning_rate=0.1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, random_state=random_state, learning_rate=0.1
        ),

        #Probabilistic
        'GaussianNB': GaussianNB()  
    }

# ─────────────────────────────────────────────────────────────────
# 2.  Cross-Validation Evaluation
# ─────────────────────────────────────────────────────────────────
 

SCORING = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy',
    'average_precision': 'average_precision',
}

def evaluate_models_cv(
    models: Dict[str,Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 10,
    random_states: int = 42,
)-> pd.DataFrame:
    """
    Stratified K-Fold cross-validation for all models.
    Returns sorted leaderboard DataFrame.
 
    NOTE: All evaluation uses TRAINING data only.
    Test set is never touched here.
    """

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_states)
    results = []
    for name,model in models.items():
        logger.info(f'CV evaluating: {name}....')
        try:
            scores = cross_validate(
                model, X_train, y_train,
                cv=cv, scoring=SCORING,
                return_train_score=False, n_jobs=-1,
            )
            row = {'Model': name}
            for metric in SCORING:
                vals = scores[f"test_{metric}"]
                row[f'{metric}_mean'] = vals.mean()
                row[f'{metric}_std'] = vals.std()
            results.append(row)

        except Exception as e:
            logger.warning(f" Failed [{name}]: {e}")

    df = pd.DataFrame(results).sort_values('roc_auc_mean', ascending = False).reset_index(drop = true)

    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────────────────────────
# 3. Hyperparameter Tuning (Optuna)
# ─────────────────────────────────────────────────────────────────
 
 def _rf_objective(trial, X,y,cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step = 50),
        'max_depth': trial.suggest_int('max_depth',3,20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2,20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt','log2',None]),
        'bootstrap': trial.suggest_categorical('bootstrap',[True,False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced',None]),
        'random_state': 42,
        'n_jobs': -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_validate(model, X,y, cv=cv, scoring='roc_auc',n_jobs=-1)
    return scores['test_roc_auc'].mean()

def _lr_objective(trial, X, y, cv):
    C = trial.suggest_float('C', 1e-4, 100.0, log=True)
    penalty = trial.suggest_categorical('penalty',['l1','l2'])
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver,
        max_iter=10000, random_state=42,
        class_weight=trial.suggest_categorical('class_weight',['balanced',None])
    )
    scores = cross_validate(model, X,y, scoring='roc_auc',n_jobs=-1)
    return scores['test_roc_auc'].mean()

def _gb_objective(trial, X,y, cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50,500, step= 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth',2,8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2,20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),
        'subsample': trial.suggest_float('subsample', 0.6,1.0),
        'max_features': trial.suggest_categorical('max_features',['sqrt','log2', None]),
        'random_state': 42,
    }
    model = GradientBoostingClassifier(**params)
    scores = cross_validate(model, X, y, cv=cv, scoring='roc_auc',n_jobs=-1)
    return scores['test_roc_auc'].mean()

OBJECTIVE_MAP = {
    'RandomForest': _rf_objective,
    'LogisticRegression': _lr_objective,
    'GradientBoosting': _gb_objective,
}

def tune_hyperparameters(
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 100,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: int = 600,
) -> tuple:
    '''
    Optuna-based hyperparameter search.
    Returns (best_params, study)
    '''
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    objective_fn = OBJECTIVE_MAP.get(model_name)

    if objective_fn is None:
        raise ValueError(
            f"No objective defined for '{model_name}'."
            f"Available: {list(OBJECTIVE_MAP.keys())}"
        )
    
    logger.info(f'Starting Optuna tuning for {model_name} | trials = {n_trials} | timeout = {timeout}s')

    study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = random_state),
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train,y_train,cv),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    logger.info(f"Best ROC-AUC (CV): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params, study

# ─────────────────────────────────────────────────────────────────
# 4. Build final model
# ─────────────────────────────────────────────────────────────────

def build_final_model(model_name: str, best_params: dict) -> Any:
    """
    Instantiates the final model with tuned hyperparameters.
    """

    model_map = {
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'GradientBoosting': GradientBoostingClassifier,
    }

    cls = model_map.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = cls(**best_params)
    logger.info(f"Final model built: {model_name}")
    return model

# ─────────────────────────────────────────────────────────────────
# 5. Ensemble / Stacking
# ─────────────────────────────────────────────────────────────────

def build_voting_ensemble(random_state: int = 42) -> VotingClassifier:
    """Soft voting ensemble of diverse model families."""
    estimators = [
        ('lr', LogisticRegression(max_iter=10000, random_state=random_state, C=1.0)),
        ('rf', RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=random_state))
        ('svc', SVC(probability=True, random_state=random_state, kernel='rbf')),
    ]

    return VotingClassifier(estimators=estimators, voting='soft',n_jobs=-1)

def best_stacking_ensemble(random_state: int=42)-> StackingClassifier:
    """
    Stacking: base learners + meta-learner (LogisticRegression).
    Powerful technique - often wins kaggle competitions.
    """

    base_learners = [
        ('lr', LogisticRegression(max_iter=10000, random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=random_state))
        ('svc', SVC(probability=True, random_state=random_state)),
        ('knn', KNeighborsClassifier(n_neighbors=7)),
        
    ]
    meta_learner = LogisticRegression(max_iter=10000, random_state=random_state)
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5, stack_method='predict_proba', n_jobs=-1,
    )

# ─────────────────────────────────────────────────────────────────
# 6. Persistence
# ─────────────────────────────────────────────────────────────────

def save_model(model, path: str = 'models/final_model.joblib') -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")

def load_model(path: str = 'models/final_model.joblib'):
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model