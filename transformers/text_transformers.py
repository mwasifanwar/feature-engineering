import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = []
        self.text_columns_ = []
        
    def fit(self, X, y=None):
        self.text_columns_ = self._detect_text_columns(X)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col in self.text_columns_:
            X_transformed[f'{col}_char_count'] = X[col].fillna('').str.len()
            X_transformed[f'{col}_word_count'] = X[col].fillna('').str.split().str.len()
            X_transformed[f'{col}_unique_words'] = X[col].fillna('').apply(
                lambda x: len(set(str(x).split())) if pd.notna(x) else 0
            )
            X_transformed[f'{col}_avg_word_length'] = X[col].fillna('').apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
            )
            X_transformed[f'{col}_sentiment_score'] = X[col].fillna('').apply(
                self._calculate_simple_sentiment
            )
            
        self.feature_names_ = X_transformed.columns.tolist()
        return X_transformed
    
    def _detect_text_columns(self, X):
        text_columns = []
        for col in X.select_dtypes(include=['object']).columns:
            sample_data = X[col].dropna().head(10)
            if len(sample_data) > 0:
                avg_length = sample_data.str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        return text_columns
    
    def _calculate_simple_sentiment(self, text):
        if not text or pd.isna(text):
            return 0
            
        text = str(text).lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        return positive_count - negative_count
    
    def get_feature_names(self):
        return self.feature_names_