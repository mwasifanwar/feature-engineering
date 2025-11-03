import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DataUtils:
    @staticmethod
    def detect_data_types(X: pd.DataFrame) -> Dict:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []
        text_cols = []
        
        for col in categorical_cols:
            if DataUtils._is_date_column(X[col]):
                date_cols.append(col)
            elif DataUtils._is_text_column(X[col]):
                text_cols.append(col)
                
        categorical_cols = [col for col in categorical_cols if col not in date_cols and col not in text_cols]
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'date': date_cols,
            'text': text_cols
        }
    
    @staticmethod
    def _is_date_column(series: pd.Series) -> bool:
        try:
            pd.to_datetime(series.head(100))
            return True
        except:
            return False
    
    @staticmethod
    def _is_text_column(series: pd.Series) -> bool:
        sample_data = series.dropna().head(10)
        if len(sample_data) == 0:
            return False
            
        avg_length = sample_data.str.len().mean()
        return avg_length > 50
    
    @staticmethod
    def handle_missing_values(X: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        X_filled = X.copy()
        
        for col in X_filled.columns:
            if X_filled[col].isna().any():
                if X_filled[col].dtype in [np.number]:
                    if strategy == 'auto':
                        X_filled[col].fillna(X_filled[col].median(), inplace=True)
                    else:
                        X_filled[col].fillna(0, inplace=True)
                else:
                    X_filled[col].fillna('missing', inplace=True)
                    
        return X_filled
    
    @staticmethod
    def remove_constant_features(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)
            elif X[col].std() < threshold:
                constant_cols.append(col)
                
        return X.drop(columns=constant_cols)