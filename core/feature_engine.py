import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class AutomatedFeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, task_type: str = 'auto', 
                 config: Optional[Dict] = None):
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or {}
        self.feature_pipeline_ = None
        self.feature_metadata_ = {}
        self.fitted_ = False
        
        from .feature_discoverer import FeatureDiscoverer
        from .feature_optimizer import FeatureOptimizer
        from .feature_validator import FeatureValidator
        
        self.discoverer = FeatureDiscoverer(target_column=target_column, 
                                          task_type=task_type)
        self.optimizer = FeatureOptimizer(target_column=target_column,
                                        task_type=task_type)
        self.validator = FeatureValidator(target_column=target_column,
                                        task_type=task_type)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None and self.target_column in X.columns:
            y = X[self.target_column]
            X = X.drop(columns=[self.target_column])
        
        discovered_features = self.discoverer.discover_features(X, y)
        optimized_features = self.optimizer.optimize_features(discovered_features, X, y)
        validation_results = self.validator.validate_features(optimized_features, X, y)
        
        self.feature_pipeline_ = self._build_pipeline(validation_results['selected_features'])
        self.feature_metadata_ = validation_results
        self.fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Must fit transformer before transforming data")
        
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
            
        return self.feature_pipeline_.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def get_feature_metadata(self) -> Dict:
        return self.feature_metadata_
    
    def _build_pipeline(self, selected_features: List[str]) -> Pipeline:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        numeric_features = [f for f in selected_features if 'cat_' not in f and 'text_' not in f]
        categorical_features = [f for f in selected_features if 'cat_' in f]
        text_features = [f for f in selected_features if 'text_' in f]
        
        transformers = []
        
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return Pipeline([
            ('feature_selector', lambda X: X[selected_features]),
            ('preprocessor', preprocessor)
        ])
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not hasattr(self, 'feature_metadata_') or 'feature_importance' not in self.feature_metadata_:
            raise ValueError("Feature importance not available. Fit the engine first.")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_metadata_['feature_importance'].keys()),
            'importance': list(self.feature_metadata_['feature_importance'].values())
        })
        return importance_df.sort_values('importance', ascending=False)