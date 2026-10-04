from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config_loader import load_config
from src.logger import get_logger
from src.playground import ordinal_pipeline

logger = get_logger()
config = load_config()

'''Feature Engineering'''
# feature-engineering
# class GarageFeatureEngineering(BaseEstimator, TransformerMixin):
#     def __init__(self, numerical_features: list[str], categorical_features: list[str], absence_features: str):
#         self.numerical_features = numerical_features
#         self.categorical_features = categorical_features
#         self.absence_features = absence_features

#     def fit(self, X : pd.date_range, y = None):

#         required_col = (
#             self.numerical_features + self.categorical_features
#         )
#         missing_col = [
#             col for col in required_col if col not in X.columns
#         ]

#         if missing_col:
#             raise ValueError(
#                 f'Missing garage columns: {missing_col}'
#             )

#     def transform(self, X : pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         # creating new feature that tells us if the garage exist or not
#         X['GarageExist'] = X['GarageYrBlt'].notna().astype(int)

#         #Filling numerical features null values with 0

#         X[self.numerical_features] = X[self.numerical_features].fillna(0)

#         #Filling cat features null values with special absence_features that is None

#         X[self.categorical_features] = X[self.categorical_features].fillna(self.absence_features)

#         return X

# # # Bsmt new feature adding

# class BsmtFeatureEngineering(BaseEstimator, TransformerMixin):
#     def __init__(self, numerical_features: list[str], categorical_features: list[str], absence_features: str):
#         self.numerical_features = numerical_features
#         self.categorical_features = categorical_features
#         self.absence_features = absence_features

#     def fit(self, X: pd.DataFrame, y = None):
#         combine_feature = self.numerical_features + self.categorical_features

#         missing_feature = [feature for feature in combine_feature if feature not in X.columns]

#         if missing_feature:
#             raise ValueError(
#                 f"Feature {missing_feature} doesn't exist"
#             )
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()

#         X['BsmtExist'] = X['BsmtCond'].notna().astype(int)

#         X[self.numerical_features] = X[self.numerical_features].fillna(0)

#         X[self.categorical_features] = X[self.categorical_features].fillna(self.absence_features)

#         return X

class StructuralFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, existence_feature: dict[str,str], numerical_features: list[str], categorical_features: list[str], absence_feature: str):
        self.existence_feature = existence_feature
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.absence_feature = absence_feature

    def fit(self, X: pd.DataFrame, y = None):
        combine_features = self.numerical_features + self.categorical_features + list[self.existence_feature.values()]

        missing_features = [col for col in combine_features if col not in X.columns]

        if missing_features:
            raise ValueError(
                f"Column {missing_features} doesn't exist"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for existence_feature, presence_feature in (self.existence_feature.items()):
            X[existence_feature] = (X[presence_feature].notna().astype(int))

        

        if self.numerical_features:
            X[self.numerical_features] = X[self.numerical_features].fillna(0)

        if self.categorical_features:
            X[self.categorical_features] = X[self.categorical_features].fillna(self.absence_feature)

        return X


class MasVnrFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X:pd.DataFrame, y = None):
        X = X.copy()
        positive_area =X [
            (X['MasVnrArea'] > 10) & (X['MasVnrType'].notna())
        ]
        self.type_area_median_ = (positive_area.groupby('MasVnrType')['MasVnrArea'].median())

        self.type_most_frequent_ = positive_area['MasVnrType'].mode().iloc[0]
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Case 1
        mask = (
            (X['MasVnrArea'] == 1) &
            (X['MasVnrType']).isna()
        )
        X.loc(mask, 'MasVnrArea') = 0
        X.loc(mask, 'MasVnrType') = 'None'
        # Case 2
        mask = (
            (X['MasVnrArea'] > 10) &
            (X['MasVnrType'].isna())
        )
        X.loc(mask, 'MasVnrType') = self.type_most_frequent_

        # Case 3

        mask = (
            (X['MasVnrArea'] == 0) &
            (X['MasVnrType'].notna())
        )

        X.loc[mask, 'MasVnrArea'] =(
             X.loc[mask, 'MasVnrType']
             .map(self.type_area_median_)
        )

        #Case 4

        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        X['MasVnrType'] = X['MasVnrType'].fillna('None')

        return X




'''Missing values'''
class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, fallback):
        self.strategy = strategy
        self.fallback = fallback

    def fit(self, X : pd.DataFrame ,y = None):
        
        if self.strategy == 'neighborhood_median':
            self.neighborhood_median_ = (
                X.groupby('Neighborhood')['LotFrontage'].median()
            )
        elif self.fallback == 'overall_median':
            self.overall_median_ = X['LotFrontage'].median()

        return self
    def transform(self, X):
        X = X.copy()
        if self.strategy == 'neighborhood_median':
            X['LotFrontage'] = X['LotFrontage'].fillna(
                X['Neighborhood'].map(self.neighborhood_median_)
            )
        elif self.fallback == 'overall_median':
            X['LotFrontage'] = X['LotFrontage'].fillna(
                self.overall_median_
            )

        return X

# other cat feature whose na will be replaced with none

categorical_imputer = SimpleImputer(
    strategy= config['preprocessor']['other_cat_feature']['strategy'],
    fill_value=config['preprocessor']['other_cat_feature']['fill_value']
)

electrical_imputer = SimpleImputer(
    strategy=config['preprocessor']['electrical']['strategy']

)

'''Encoding the categoricals'''

nominal_encoder = OneHotEncoder(
    handle_unknown= 'ignore',
    sparse_output= False
)

'''ordinal encoding'''

quality_order = [
    "None", "Po", "Fa", "TA", "Gd", "Ex"
]

exposure_order = [
    "None", "No", "Mn", "Av", "Gd"
]

basement_finish_order = [
    "None", "Unf", "Lwq", "Rec", "BLQ", "ALQ", "GLQ"
]

garage_finish_order = [
    "None", "Unf", "RFn", "Fin"
]

fence_order = [
    "None", "MnWw", "GdWo", "MnPrv", "GdPrv"
]

quality_order_encoder = OrdinalEncoder(
    categories=len(config['preprocessor']['ordinal']['quality']) * [quality_order],
    handle_unknown='use_encoded_value',
    unknown_value= -1
)

exposure_order_encoder = OrdinalEncoder(
    categories= [exposure_order] * len(config['preprocessor']['ordinal']['exposure']),
    handle_unknown='use_encoded_value',
    unknown_value= -1
)

basement_finish_order_encoder = OrdinalEncoder(
    categories= [basement_finish_order] * len(config['preprocessor']['ordinal']['basement']),
    handle_unknown='use_encoded_value',
    unknown_value= -1
)

garage_finish_order_encoder = OrdinalEncoder(
    categories= [garage_finish_order] * len(config['preprocessor']['ordinal']['garage']),
    handle_unknown='use_encoded_value',
    unknown_value= -1
)

fence_order_encoder = OrdinalEncoder(
    categories= [fence_order] * len(config['preprocessor']['ordinal']['fence'])
)



'''Skewness'''

class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], method: str):
        self.features = features
        self.method = method

    def fit(self, X : pd.DataFrame, y = None):
        missing_col = [col for col in self.features if col not in X.columns]

        if missing_col:
            raise ValueError(f"Columns {missing_col} doesn't exist")
        if self.method != 'log1p':
            raise ValueError(f"Unsupported method: {self.method}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.features] = np.log1p(X[self.features])

        return X


'''Column Transformer'''

ordinal_transformer = ColumnTransformer(
    transformers=[
        (
            'quality',
            quality_order_encoder,
            config['preprocessor']['ordinal']['quality'],
        ),
        (
            'exposure',
            exposure_order_encoder,
            config['preprocessor']['ordinal']['exposure'],
        ),
        (
            'basement',
            basement_finish_order_encoder,
            config['preprocessor']['ordinal']['basement'],
        ),
        (
            'garage',
            garage_finish_order_encoder,
            config['preprocessor']['ordinal']['garage'],
        ),
        (
            'fence',
            fence_order_encoder,
            config['preprocessor']['ordinal']['fence']
        )
    ],
    remainder='passthrough'
)


# ===============================================
# Loading preprocessor
# ===============================================

def build_preprocessor(config : dict) -> Pipeline:


    config_preprocessor = config['preprocessor']

    #----------------------------------
    # 1. structural feature engineering
    # ---------------------------------
    garage_config = config_preprocessor['garage']
    basement_config = config_preprocessor['bsmt']
    existence_config = config_preprocessor['existence_feature']
    absence_config = config_preprocessor['absence_feature']

    structural_numerical_features = (garage_config['numerical_features'] + basement_config['numerical_features'])
    structural_categorical_features = (garage_config['categorical_features'] + basement_config['categorical_features'])

    structural_transformer = StructuralFeatureEngineer(
        existence_feature= existence_config,
        numerical_features=structural_numerical_features,
        categorical_features= structural_categorical_features,
        absence_feature= absence_config
    )

    #------------------------------
    # 2. LotFrontage
    #-----------------------------

    lotfrontage_config = config_preprocessor['lotfrontage']
    lotfrontage_transformer = LotFrontageImputer(
        strategy=lotfrontage_config['strategy'],
        fallback=lotfrontage_config['fallback']
    )

    #--------------------------------
    # 3. Skewness transformer
    #-------------------------------

    skewness_config = config_preprocessor['skewness']
    skewness_transformer = SkewnessTransformer(
        features=skewness_config['features'], method= skewness_config['method']
    )

    #--------------------------
    # 4. Ordinal transformer
    #--------------------------

    ordinal_config = config_preprocessor['ordinal']
    ordinal_quality_features = ordinal_config['quality']['features']
    ordinal_exposure_features = ordinal_config['exposure']['features']
    ordinal_basement_features = ordinal_config['basement']['features']
    ordinal_garage_features = ordinal_config['garage']['features']
    ordinal_fence_features = ordinal_config['fence']['features']

    ordinal_quality_features_pipeline = ordinal_pipeline(
        features=ordinal_quality_features, order= ordinal_config['quality']['order']
    )
    ordinal_exposure_features_pipeline = ordinal_pipeline(
        features= ordinal_exposure_features, order= ordinal_config['exposure']['order']
    )
    ordinal_basement_features_pipeline = ordinal_pipeline(
        features= ordinal_basement_features, order= ordinal_config['basement']['order']
    )
    ordinal_garage_features_pipeline = ordinal_pipeline(
        features= ordinal_garage_features, order= ordinal_config['garage']['order']
    )
    ordinal_fence_features_pipeline = ordinal_pipeline(
        features= ordinal_fence_features, order= ordinal_config['fence']['order']
    )


    #------------------------------
    # 5. Nominal/categorical transformer
    #------------------------------

    nominal_config = config_preprocessor['other_cat_features']
    nominal_config_features = nominal_config['features']
    nominal_pipeline = Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(
                    strategy= nominal_config['strategy'],
                    fill_value='None',
                )
            ),
            (
                'encoder',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                ),
            ),
        ]
    )

    #-------------------------
    # 6.Electric
    #------------------------

    electrical_config = config_preprocessor['electrical']
    electric_impute = SimpleImputer(
        strategy=electrical_config['strategy'],
    )

    #-----------------------------
    # 7. Final columnTransformers
    #-----------------------------

    column_transformer = ColumnTransformer(
        transformers=[
            (
                'skewed_numericals',
                skewness_transformer,
                skewness_config['features']
            ),
            (
                'ordinal_quality',
                ordinal_quality_features_pipeline,
                ordinal_quality_features,
            ),
            (
                'ordinal_exposure',
                ordinal_exposure_features_pipeline,
                ordinal_exposure_features,
            ),
            (
                'ordinal_basement',
                ordinal_basement_features_pipeline,
                ordinal_basement_features,
            ),
            (
                'ordinal_garage',
                ordinal_garage_features_pipeline,
                ordinal_garage_features,
            ),
            (
                'ordinal_fence',
                ordinal_fence_features_pipeline,
                ordinal_fence_features,
            ),
            (
                'nominal',
                nominal_pipeline,
                nominal_config_features
            ),
            (
                'electrical',
                electric_impute,
                electrical_config['feature']
            )
        ],
        remainder='passthrough',
    )


    #--------------------------------------
    # 8. Complete preprocessing pipeline
    #-------------------------------------

    preprocessing_pipeline = Pipeline(
        steps=[
            (
                'structural',
                structural_transformer,
            ),
            (
                'masvnr',
                MasVnrFeatureEngineer(),
            ),
            (
                'lotfrontage',
                lotfrontage_transformer,
            ),
            (
                'columntransformer',
                column_transformer,
            ),
        ]
    )

    return preprocessing_pipeline


preprocessor = build_preprocessor(config)











