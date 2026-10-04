# dummy models
from matplotlib.pylab import Any
from sklearn.dummy import DummyRegressor
# linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# distance-based models
from sklearn.neighbors import KNeighborsRegressor
# tree-based models
from sklearn.tree import (
    DecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,   
)
# other models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



# Local imports
import __main__

from src.logger import get_logger
from src.config_loader import load_config


config = load_config()
logger = get_logger(__name__)

def get_baseline_models(config : dict) -> dict[str, Any]:
    """
    Returns a baseline model for regression tasks.
    """
    return {
        'DummyRegressor (mean)': DummyRegressor(strategy='mean'),
        'DummyRegressor (median)': DummyRegressor(strategy='median'),
        'LinearRegression': LinearRegression(n_jobs=config['models']['linear_models']['LinearRegression']['n_jobs']),
    }

def get_candidate_models(config : dict) -> dict[str, Any]:
    """
    Returns a dictionary of candidate models for regression tasks.
    """
    return {
        # Linear Models
        'Ridge': Ridge(max_iter=config['models']['linear_models']['Ridge']['max_iter'], random_state=config['random_state']),
        'Lasso': Lasso(alpha=config['models']['linear_models']['Lasso']['alpha'], random_state=config['random_state']),
        'ElasticNet': ElasticNet(alpha=config['models']['linear_models']['ElasticNet']['alpha'], l1_ratio=config['models']['linear_models']['ElasticNet']['l1_ratio'], random_state=config['random_state']),

        # Distance-based Models
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=config['models']['distance_based_models']['KNeighborsRegressor']['n_neighbors']),

        # Tree-based Models
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=config['models']['tree_based_models']['DecisionTreeRegressor']['max_depth'],
                                                      min_samples_split=config['models']['tree_based_models']['DecisionTreeRegressor']['min_samples_split'],
                                                      min_samples_leaf=config['models']['tree_based_models']['DecisionTreeRegressor']['min_samples_leaf'],
                                                      random_state=config['random_state']),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=config['models']['tree_based_models']['RandomForestRegressor']['n_estimators'],
                                                      max_depth=config['models']['tree_based_models']['RandomForestRegressor']['max_depth'],
                                                      min_samples_split=config['models']['tree_based_models']['RandomForestRegressor']['min_samples_split'],
                                                      min_samples_leaf=config['models']['tree_based_models']['RandomForestRegressor']['min_samples_leaf'],
                                                      n_jobs=config['models']['tree_based_models']['RandomForestRegressor']['n_jobs'],
                                                      random_state=config['random_state']),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=config['models']['tree_based_models']['GradientBoostingRegressor']['n_estimators'],
                                                               learning_rate=config['models']['tree_based_models']['GradientBoostingRegressor']['learning_rate'],
                                                               max_depth=config['models']['tree_based_models']['GradientBoostingRegressor']['max_depth'],
                                                               random_state=config['random_state']),
        'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=config['models']['tree_based_models']['ExtraTreesRegressor']['n_estimators'],
                                                   max_depth=config['models']['tree_based_models']['ExtraTreesRegressor']['max_depth'],random_state=config['models']['tree_based_models']['ExtraTreesRegressor']['random_state'],),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(max_iter=config['models']['tree_based_models']['HistGradientBoostingRegressor']['max_iter'],
                                                                      learning_rate=config['models']['tree_based_models']['HistGradientBoostingRegressor']['learning_rate'],
                                                                      max_depth=config['models']['tree_based_models']['HistGradientBoostingRegressor']['max_depth']),

        # Other Models
        'XGBRegressor': XGBRegressor(n_estimators=config['models']['other_models']['XGBRegressor']['n_estimators'],
                                     learning_rate=config['models']['other_models']['XGBRegressor']['learning_rate'],
                                     max_depth=config['models']['other_models']['XGBRegressor']['max_depth']),
        'LGBMRegressor': LGBMRegressor(n_estimators=config['models']['other_models']['LGBMRegressor']['n_estimators'],
                                       learning_rate=config['models']['other_models']['LGBMRegressor']['learning_rate'],
                                       max_depth=config['models']['other_models']['LGBMRegressor']['max_depth']),
        'CatBoostRegressor': CatBoostRegressor(iterations=config['models']['other_models']['CatBoostRegressor']['iterations'],
                                                learning_rate=config['models']['other_models']['CatBoostRegressor']['learning_rate'],
                                                depth=config['models']['other_models']['CatBoostRegressor']['depth'],
                                                verbose=0),
    }







