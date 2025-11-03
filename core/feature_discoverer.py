import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import re

class FeatureDiscoverer:
    def __init__(self, target_column: str, task_type: str = 'auto'):
        self.target_column = target_column
        self.task_type = task_type
        self.generated_features_ = []
        
    def discover_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[str]]:
        feature_categories = {}
        
        feature_categories['statistical'] = self._generate_statistical_features(X)
        feature_categories['interactions'] = self._generate_interaction_features(X)
        feature_categories['transformations'] = self._generate_transformation_features(X)
        feature_categories['temporal'] = self._generate_temporal_features(X)
        feature_categories['categorical'] = self._generate_categorical_features(X)
        feature_categories['text'] = self._generate_text_features(X)
        
        all_features = []
        for category, features in feature_categories.items():
            all_features.extend(features)
            
        self.generated_features_ = all_features
        return feature_categories
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> List[str]:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        statistical_features = []
        
        for col in numeric_cols:
            statistical_features.extend([
                f'{col}_mean',
                f'{col}_std',
                f'{col}_skew',
                f'{col}_kurtosis',
                f'{col}_q25',
                f'{col}_q75',
                f'{col}_range'
            ])
            
        return statistical_features
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> List[str]:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        interaction_features = []
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interaction_features.extend([
                    f'{col1}_x_{col2}',
                    f'{col1}_div_{col2}',
                    f'{col1}_plus_{col2}',
                    f'{col1}_minus_{col2}'
                ])
                
        return interaction_features
    
    def _generate_transformation_features(self, X: pd.DataFrame) -> List[str]:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        transformation_features = []
        
        for col in numeric_cols:
            transformation_features.extend([
                f'{col}_log',
                f'{col}_sqrt',
                f'{col}_square',
                f'{col}_cube',
                f'{col}_reciprocal'
            ])
            
        return transformation_features
    
    def _generate_temporal_features(self, X: pd.DataFrame) -> List[str]:
        temporal_features = []
        date_columns = self._detect_date_columns(X)
        
        for col in date_columns:
            temporal_features.extend([
                f'{col}_year',
                f'{col}_month', 
                f'{col}_day',
                f'{col}_dayofweek',
                f'{col}_quarter',
                f'{col}_is_weekend'
            ])
            
        return temporal_features
    
    def _generate_categorical_features(self, X: pd.DataFrame) -> List[str]:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_features = []
        
        for col in categorical_cols:
            categorical_features.extend([
                f'{col}_freq',
                f'{col}_encoded',
                f'{col}_target_encoded'
            ])
            
        return categorical_features
    
    def _generate_text_features(self, X: pd.DataFrame) -> List[str]:
        text_cols = self._detect_text_columns(X)
        text_features = []
        
        for col in text_cols:
            text_features.extend([
                f'{col}_char_count',
                f'{col}_word_count',
                f'{col}_unique_words',
                f'{col}_avg_word_length',
                f'{col}_sentiment_score'
            ])
            
        return text_features
    
    def _detect_date_columns(self, X: pd.DataFrame) -> List[str]:
        date_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col].head(100))
                    date_columns.append(col)
                except:
                    continue
        return date_columns
    
    def _detect_text_columns(self, X: pd.DataFrame) -> List[str]:
        text_columns = []
        for col in X.select_dtypes(include=['object']).columns:
            sample_data = X[col].dropna().head(100)
            if len(sample_data) > 0:
                avg_length = sample_data.str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
        return text_columns