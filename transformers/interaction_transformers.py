import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class InteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = []
        self.interaction_pairs_ = []
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.interaction_pairs_ = []
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                self.interaction_pairs_.append((col1, col2))
                
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col1, col2 in self.interaction_pairs_:
            X_transformed[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            X_transformed[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
            X_transformed[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
            X_transformed[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
            
        self.feature_names_ = X_transformed.columns.tolist()
        return X_transformed
    
    def get_feature_names(self):
        return self.feature_names_