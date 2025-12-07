import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing import preprocess_pipeline
from feature_engineering import feature_engineering_pipeline
from models import train_regression_model, train_classification_model, save_model
from evaluation import generate_evaluation_report


def main(args):
    print("="*70)
    print("STUDENT PERFORMANCE / DROPOUT PREDICTION WITH ML")
    print("="*70)
    
    print("\n[Step 1/5] Loading and preprocessing data...")
    data = preprocess_pipeline(
        data_dir=args.data_dir,
        combine_datasets=args.combine_datasets,
        dropout_threshold=args.dropout_threshold,
        test_size=args.test_size
    )
    
    print("\n[Step 2/5] Feature engineering...")
    feat_reg = feature_engineering_pipeline(
        data['regression']['X_train'],
        data['regression']['X_test'],
        data['regression']['y_train'],
        create_derived=args.create_derived_features,
        encode_categorical=True,
        scale_features_flag=False,
        feature_selection=args.feature_selection,
        top_k=args.top_k_features
    )
    
    feat_clf = feature_engineering_pipeline(
        data['classification']['X_train'],
        data['classification']['X_test'],
        data['classification']['y_train'],
        create_derived=args.create_derived_features,
        encode_categorical=True,
        scale_features_flag=False,
        feature_selection=args.feature_selection,
        top_k=args.top_k_features
    )
    
    print("\n[Step 3/5] Training models...")
    reg_model, reg_results = train_regression_model(
        feat_reg['X_train'],
        data['regression']['y_train'],
        feat_reg['X_test'],
        data['regression']['y_test'],
        use_grid_search=args.hyperparameter_tuning,
        cv=args.cv_folds,
        random_state=args.random_state
    )
    
    clf_model, clf_results = train_classification_model(
        feat_clf['X_train'],
        data['classification']['y_train'],
        feat_clf['X_test'],
        data['classification']['y_test'],
        use_grid_search=args.hyperparameter_tuning,
        cv=args.cv_folds,
        random_state=args.random_state
    )
    
    if args.save_models:
        print("\n[Step 4/5] Saving models...")
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        save_model(reg_model, models_dir / 'regression_model.pkl', 'regression')
        save_model(clf_model, models_dir / 'classification_model.pkl', 'classification')
    
    print("\n[Step 5/5] Evaluating models...")
    eval_data = {
        'regression': {
            'X_test': feat_reg['X_test'],
            'y_test': data['regression']['y_test']
        },
        'classification': {
            'X_test': feat_clf['X_test'],
            'y_test': data['classification']['y_test']
        }
    }
    generate_evaluation_report(reg_model, clf_model, eval_data, results_dir=args.results_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.results_dir}/")
    print(f"Models saved to: models/")
    print("\nKey Results:")
    print(f"  Regression - RMSE: {reg_results.get('test_rmse', 'N/A'):.4f}, "
          f"R²: {reg_results.get('test_r2', 'N/A'):.4f}")
    print(f"  Classification - Accuracy: {clf_results.get('test_accuracy', 'N/A'):.4f}, "
          f"ROC-AUC: {clf_results.get('test_roc_auc', 'N/A'):.4f}")
    print("\nEvaluation plots and metrics are available in the results directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Student Performance / Dropout Prediction with ML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='student',
                       help='Directory containing dataset files')
    parser.add_argument('--combine-datasets', action='store_true', default=True,
                       help='Combine math and Portuguese datasets')
    parser.add_argument('--dropout-threshold', type=int, default=10,
                       help='Grade threshold for dropout classification (G3 < threshold)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of test set')
    parser.add_argument('--create-derived-features', action='store_true', default=True,
                       help='Create derived features (grade averages, improvements, etc.)')
    parser.add_argument('--feature-selection', action='store_true', default=False,
                       help='Perform feature selection')
    parser.add_argument('--top-k-features', type=int, default=30,
                       help='Number of top features to select')
    parser.add_argument('--hyperparameter-tuning', action='store_true', default=False,
                       help='Perform hyperparameter tuning (slower but better performance)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Save trained models to disk')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results and visualizations')
    
    args = parser.parse_args()
    
    main(args)

