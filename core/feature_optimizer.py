import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FeatureOptimizer:
    def __init__(self, target_column: str, task_type: str = 'auto',
                 max_features: int = 100, correlation_threshold: float = 0.95):
        self.target_column = target_column
        self.task_type = task_type
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.optimized_features_ = []
        
    def optimize_features(self, feature_categories: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        all_features = []
        for category_features in feature_categories.values():
            all_features.extend(category_features)
            
        importance_scores = self._calculate_feature_importance(all_features, X, y)
        stability_scores = self._calculate_feature_stability(all_features, X, y)
        correlation_matrix = self._calculate_feature_correlation(all_features, X)
        
        selected_features = self._select_features(
            all_features, importance_scores, stability_scores, correlation_matrix
        )
        
        optimization_results = {
            'selected_features': selected_features,
            'importance_scores': importance_scores,
            'stability_scores': stability_scores,
            'correlation_matrix': correlation_matrix
        }
        
        self.optimized_features_ = selected_features
        return optimization_results
    
    def _calculate_feature_importance(self, features: List[str], X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        importance_scores = {}
        
        if self.task_type in ['regression', 'auto']:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            score_func = f_regression
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            score_func = f_classif
            
        try:
            selector = SelectKBest(score_func=score_func, k='all')
            selector.fit(X[features], y)
            
            for i, feature in enumerate(features):
                importance_scores[feature] = selector.scores_[i]
        except:
            for feature in features:
                importance_scores[feature] = 0.0
                
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        for feature in importance_scores:
            importance_scores[feature] /= max_score
            
        return importance_scores
    
    def _calculate_feature_stability(self, features: List[str], X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        stability_scores = {}
        n_splits = 3
        
        for feature in features:
            try:
                scores = []
                for _ in range(n_splits):
                    sample_indices = np.random.choice(len(X), size=len(X)//2, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_sample = y.iloc[sample_indices]
                    
                    if self.task_type in ['regression', 'auto']:
                        model = RandomForestRegressor(n_estimators=20, random_state=42)
                        score_func = f_regression
                    else:
                        model = RandomForestClassifier(n_estimators=20, random_state=42)
                        score_func = f_classif
                        
                    selector = SelectKBest(score_func=score_func, k=1)
                    selector.fit(X_sample[[feature]], y_sample)
                    scores.append(selector.scores_[0])
                
                stability_scores[feature] = 1 - (np.std(scores) / (np.mean(scores) + 1e-8))
            except:
                stability_scores[feature] = 0.0
                
        return stability_scores
    
    def _calculate_feature_correlation(self, features: List[str], X: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_features = [f for f in features if f in X.select_dtypes(include=[np.number]).columns]
            return X[numeric_features].corr()
        except:
            return pd.DataFrame()
    
    def _select_features(self, features: List[str], importance_scores: Dict, 
                        stability_scores: Dict, correlation_matrix: pd.DataFrame) -> List[str]:
        feature_scores = []
        
        for feature in features:
            importance = importance_scores.get(feature, 0)
            stability = stability_scores.get(feature, 0)
            
            overall_score = 0.7 * importance + 0.3 * stability
            feature_scores.append((feature, overall_score))
        
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = []
        for feature, score in feature_scores:
            if len(selected_features) >= self.max_features:
                break
                
            if score > 0.01:
                selected_features.append(feature)
                
        return selected_features[:self.max_features]