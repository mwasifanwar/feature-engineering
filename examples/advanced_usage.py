import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AutomatedFeatureEngine
from config import FeatureConfig

def advanced_usage_example():
    print("=== Automated Feature Engineering - Advanced Usage Example ===")
    
    np.random.seed(42)
    n_samples = 2000
    
    data = pd.DataFrame({
        'price': np.random.lognormal(3, 1, n_samples),
        'quantity': np.random.poisson(50, n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'purchase_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'review_text': np.random.choice([
            'amazing product fast delivery', 
            'poor quality would not recommend',
            'good value for money',
            'excellent customer service'
        ], n_samples),
        'sales': np.random.gamma(2, 2, n_samples)
    })
    
    config = FeatureConfig(
        max_features=50,
        feature_interactions=True,
        temporal_features=True,
        text_features=True,
        feature_selection_method='multi_objective',
        stability_threshold=0.8
    )
    
    engine = AutomatedFeatureEngine(
        target_column='sales',
        task_type='regression',
        config=config.to_dict()
    )
    
    feature_matrix = engine.fit_transform(data)
    
    metadata = engine.get_feature_metadata()
    
    print("Advanced Feature Engineering Results:")
    print(f"Original features: {len(data.columns)}")
    print(f"Generated features: {metadata['final_feature_count']}")
    print(f"Performance improvement: {metadata['performance_metrics']['improvement']:.4f}")
    print(f"Feature stability scores: {np.mean(list(metadata['stability_analysis'].values())):.4f}")
    
    return engine, feature_matrix, metadata

if __name__ == "__main__":
    engine, features, metadata = advanced_usage_example()