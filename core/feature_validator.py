import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class FeatureValidator:
    def __init__(self, target_column: str, task_type: str = 'auto',
                 cv_folds: int = 5, validation_metric: str = 'auto'):
        self.target_column = target_column
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.validation_metric = validation_metric
        self.validation_results_ = {}
        
    def validate_features(self, optimization_results: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        selected_features = optimization_results['selected_features']
        
        performance_metrics = self._evaluate_feature_performance(selected_features, X, y)
        stability_analysis = self._analyze_feature_stability(selected_features, X, y)
        redundancy_check = self._check_feature_redundancy(selected_features, X)
        
        validation_results = {
            'selected_features': selected_features,
            'performance_metrics': performance_metrics,
            'stability_analysis': stability_analysis,
            'redundancy_check': redundancy_check,
            'feature_importance': optimization_results['importance_scores'],
            'final_feature_count': len(selected_features)
        }
        
        self.validation_results_ = validation_results
        return validation_results
    
    def _evaluate_feature_performance(self, features: List[str], X: pd.DataFrame, y: pd.Series) -> Dict:
        if len(features) == 0:
            return {'baseline_score': 0, 'feature_score': 0, 'improvement': 0}
            
        if self.task_type in ['regression', 'auto']:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            baseline_model = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'neg_mean_squared_error'
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        
        try:
            baseline_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:10]
            if len(baseline_cols) > 0:
                baseline_scores = cross_val_score(baseline_model, X[baseline_cols], y, 
                                                cv=self.cv_folds, scoring=scoring)
                baseline_score = np.mean(baseline_scores)
            else:
                baseline_score = 0
                
            feature_scores = cross_val_score(model, X[features], y, 
                                           cv=self.cv_folds, scoring=scoring)
            feature_score = np.mean(feature_scores)
            
            improvement = feature_score - baseline_score
            
        except:
            baseline_score = 0
            feature_score = 0
            improvement = 0
            
        return {
            'baseline_score': baseline_score,
            'feature_score': feature_score,
            'improvement': improvement
        }
    
    def _analyze_feature_stability(self, features: List[str], X: pd.DataFrame, y: pd.Series) -> Dict:
        stability_results = {}
        n_iterations = 5
        
        feature_presence = {feature: 0 for feature in features}
        
        for iteration in range(n_iterations):
            train_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            
            try:
                if self.task_type in ['regression', 'auto']:
                    model = RandomForestRegressor(n_estimators=30, random_state=42)
                    score_func = 'f_regression'
                else:
                    model = RandomForestClassifier(n_estimators=30, random_state=42)
                    score_func = 'f_classif'
                    
                from sklearn.feature_selection import SelectKBest
                selector = SelectKBest(score_func=score_func, k=min(20, len(features)))
                selector.fit(X_train[features], y_train)
                
                selected_indices = selector.get_support(indices=True)
                for idx in selected_indices:
                    if idx < len(features):
                        feature_presence[features[idx]] += 1
                        
            except:
                continue
                
        for feature in features:
            stability_results[feature] = feature_presence[feature] / n_iterations
            
        return stability_results
    
    def _check_feature_redundancy(self, features: List[str], X: pd.DataFrame) -> Dict:
        redundancy_results = {}
        
        try:
            numeric_features = [f for f in features if f in X.select_dtypes(include=[np.number]).columns]
            correlation_matrix = X[numeric_features].corr().abs()
            
            for feature in numeric_features:
                correlated_features = correlation_matrix[feature][correlation_matrix[feature] > 0.9]
                redundancy_results[feature] = len(correlated_features) - 1
                
        except:
            for feature in features:
                redundancy_results[feature] = 0
                
        return redundancy_results