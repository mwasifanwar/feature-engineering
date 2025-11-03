import argparse
import pandas as pd
from core import AutomatedFeatureEngine
from examples.basic_usage import basic_usage_example
from examples.advanced_usage import advanced_usage_example

def main():
    parser = argparse.ArgumentParser(description="Automated Feature Engineering Engine")
    parser.add_argument('--mode', choices=['demo', 'train', 'process'], default='demo',
                       help='Operation mode: demo, train, or process')
    parser.add_argument('--input', type=str, help='Input data file path')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--task', choices=['regression', 'classification'], default='auto',
                       help='Machine learning task type')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Automated Feature Engineering Demo")
        basic_usage_example()
        
    elif args.mode == 'process' and args.input and args.target:
        print(f"Processing data from {args.input}")
        
        data = pd.read_csv(args.input)
        engine = AutomatedFeatureEngine(
            target_column=args.target,
            task_type=args.task
        )
        
        features = engine.fit_transform(data)
        
        if args.output:
            features.to_csv(args.output, index=False)
            print(f"Features saved to {args.output}")
        else:
            print("Transformed features:")
            print(features.head())
            
    else:
        print("Please provide required arguments for processing mode")
        print("Usage: python main.py --mode process --input data.csv --target target_column")

if __name__ == "__main__":
    main()