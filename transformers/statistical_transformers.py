import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class StatisticalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            X_transformed[f'{col}_mean'] = X[col].mean()
            X_transformed[f'{col}_std'] = X[col].std()
            X_transformed[f'{col}_skew'] = stats.skew(X[col].fillna(0))
            X_transformed[f'{col}_kurtosis'] = stats.kurtosis(X[col].fillna(0))
            X_transformed[f'{col}_q25'] = X[col].quantile(0.25)
            X_transformed[f'{col}_q75'] = X[col].quantile(0.75)
            X_transformed[f'{col}_range'] = X[col].max() - X[col].min()
            
        self.feature_names_ = X_transformed.columns.tolist()
        return X_transformed
    
    def get_feature_names(self):
        return self.feature_names_