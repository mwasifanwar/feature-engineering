import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Dict

class FeaturePipeline:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.pipeline_ = None
        self.feature_names_ = []
        
    def build_pipeline(self, feature_categories: Dict, X: pd.DataFrame) -> Pipeline:
        numeric_features = []
        categorical_features = []
        
        for category, features in feature_categories.items():
            for feature in features:
                if feature in X.columns:
                    if X[feature].dtype in [np.number]:
                        numeric_features.append(feature)
                    else:
                        categorical_features.append(feature)
        
        transformers = []
        
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
            
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
            
        preprocessor = ColumnTransformer(transformers=transformers)
        
        self.pipeline_ = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        self.feature_names_ = numeric_features + categorical_features
        return self.pipeline_
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.pipeline_ is None:
            self.build_pipeline({'default': X.columns.tolist()}, X)
            
        self.pipeline_.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise ValueError("Pipeline not built. Call fit first.")
            
        transformed = self.pipeline_.transform(X)
        
        if hasattr(self.pipeline_.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = self.pipeline_.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = [f'feature_{i}' for i in range(transformed.shape[1])]
            
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)