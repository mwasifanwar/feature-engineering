import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = []
        self.date_columns_ = []
        
    def fit(self, X, y=None):
        self.date_columns_ = self._detect_date_columns(X)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col in self.date_columns_:
            try:
                dates = pd.to_datetime(X[col])
                X_transformed[f'{col}_year'] = dates.dt.year
                X_transformed[f'{col}_month'] = dates.dt.month
                X_transformed[f'{col}_day'] = dates.dt.day
                X_transformed[f'{col}_dayofweek'] = dates.dt.dayofweek
                X_transformed[f'{col}_quarter'] = dates.dt.quarter
                X_transformed[f'{col}_is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
            except:
                continue
                
        self.feature_names_ = X_transformed.columns.tolist()
        return X_transformed
    
    def _detect_date_columns(self, X):
        date_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col].head(10))
                    date_columns.append(col)
                except:
                    continue
        return date_columns
    
    def get_feature_names(self):
        return self.feature_names_