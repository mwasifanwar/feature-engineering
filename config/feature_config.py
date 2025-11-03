from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FeatureConfig:
    max_features: int = 100
    correlation_threshold: float = 0.95
    importance_threshold: float = 0.01
    stability_threshold: float = 0.7
    cv_folds: int = 5
    
    feature_interactions: bool = True
    polynomial_degree: int = 2
    temporal_features: bool = True
    text_features: bool = True
    categorical_features: bool = True
    
    feature_selection_method: str = 'multi_objective'
    validation_strategy: str = 'cross_validation'
    optimization_metric: str = 'auto'
    
    def to_dict(self) -> Dict:
        return {
            'max_features': self.max_features,
            'correlation_threshold': self.correlation_threshold,
            'importance_threshold': self.importance_threshold,
            'stability_threshold': self.stability_threshold,
            'cv_folds': self.cv_folds,
            'feature_interactions': self.feature_interactions,
            'polynomial_degree': self.polynomial_degree,
            'temporal_features': self.temporal_features,
            'text_features': self.text_features,
            'categorical_features': self.categorical_features,
            'feature_selection_method': self.feature_selection_method,
            'validation_strategy': self.validation_strategy,
            'optimization_metric': self.optimization_metric
        }