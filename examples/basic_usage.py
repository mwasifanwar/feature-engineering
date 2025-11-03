import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AutomatedFeatureEngine

def basic_usage_example():
    print("=== Automated Feature Engineering - Basic Usage Example ===")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'text_col': np.random.choice(['excellent product', 'bad experience', 'good quality'], n_samples),
        'target': np.random.normal(0, 1, n_samples)
    })
    
    print("Original data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    
    engine = AutomatedFeatureEngine(
        target_column='target',
        task_type='regression'
    )
    
    feature_matrix = engine.fit_transform(data)
    
    print("Transformed data shape:", feature_matrix.shape)
    print("Generated features:", len(engine.get_feature_metadata()['selected_features']))
    
    feature_importance = engine.get_feature_importance()
    print("Top 10 features by importance:")
    print(feature_importance.head(10))
    
    return engine, feature_matrix

if __name__ == "__main__":
    engine, features = basic_usage_example()