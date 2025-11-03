import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

class ValidationUtils:
    @staticmethod
    def calculate_feature_quality(X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict:
        quality_metrics = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if task_type == 'regression':
                correlation = abs(X[col].corr(y))
                quality_metrics[col] = correlation
            else:
                from scipy.stats import f_oneway
                try:
                    groups = [X[col][y == cls] for cls in y.unique()]
                    f_stat, _ = f_oneway(*groups)
                    quality_metrics[col] = f_stat
                except:
                    quality_metrics[col] = 0
                    
        return quality_metrics
    
    @staticmethod
    def validate_feature_stability(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                 features: List[str]) -> Dict:
        stability_scores = {}
        
        for feature in features:
            if feature in X_train.columns and feature in X_test.columns:
                train_stats = {
                    'mean': X_train[feature].mean(),
                    'std': X_train[feature].std(),
                    'min': X_train[feature].min(),
                    'max': X_train[feature].max()
                }
                
                test_stats = {
                    'mean': X_test[feature].mean(),
                    'std': X_test[feature].std(),
                    'min': X_test[feature].min(),
                    'max': X_test[feature].max()
                }
                
                mean_diff = abs(train_stats['mean'] - test_stats['mean']) / (train_stats['std'] + 1e-8)
                std_ratio = min(train_stats['std'], test_stats['std']) / max(train_stats['std'], test_stats['std'])
                
                stability_score = (1 - mean_diff) * std_ratio
                stability_scores[feature] = max(0, stability_score)
            else:
                stability_scores[feature] = 0
                
        return stability_scores