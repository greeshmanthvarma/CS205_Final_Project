import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from pathlib import Path


def evaluate_regression_model(model, X_test, y_test, save_path=None):
    """
    Evaluate regression model and generate metrics.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained regression model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    save_path : str or Path, optional
        Directory to save evaluation results
    
    Returns:
    --------
    results : dict
        Dictionary with evaluation metrics
    """
    print("\nEvaluating Regression Model...")
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    non_zero_mask = y_test != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
    else:
        mape = np.nan
    residuals = y_test - y_pred
    
    results = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'predictions': y_pred,
        'residuals': residuals
    }
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Grade (G3)', fontsize=12)
        plt.ylabel('Predicted Grade (G3)', fontsize=12)
        plt.title(f'Prediction vs Actual (R² = {r2:.3f})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'regression_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Grade', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Residuals Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'regression_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', lw=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'regression_residuals_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {save_path}")
    
    return results


def evaluate_classification_model(model, X_test, y_test, save_path=None):
    """
    Evaluate classification model and generate metrics.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained classification model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    save_path : str or Path, optional
        Directory to save evaluation results
    
    Returns:
    --------
    results : dict
        Dictionary with evaluation metrics
    """
    print("\nEvaluating Classification Model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'Confusion Matrix': cm,
        'predictions': y_pred,
        'prediction_proba': y_pred_proba
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Dropout', 'Dropout'],
                   yticklabels=['No Dropout', 'Dropout'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'classification_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='No Dropout', color='green')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Dropout', color='red')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'classification_probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        report = classification_report(y_test, y_pred, 
                                       target_names=['No Dropout', 'Dropout'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(save_path / 'classification_report.csv')
        
        print(f"Evaluation plots saved to {save_path}")
    
    return results


def compare_models(results_reg, results_clf, save_path=None):
    """
    Create comparison visualization of both models.
    
    Parameters:
    -----------
    results_reg : dict
        Regression model results
    results_clf : dict
        Classification model results
    save_path : str or Path, optional
        Directory to save comparison plots
    """
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create metrics comparison
        metrics_data = {
            'Regression': {
                'RMSE': results_reg.get('RMSE', 0),
                'MAE': results_reg.get('MAE', 0),
                'R²': results_reg.get('R²', 0)
            },
            'Classification': {
                'Accuracy': results_clf.get('Accuracy', 0),
                'F1': results_clf.get('F1', 0),
                'ROC-AUC': results_clf.get('ROC-AUC', 0)
            }
        }
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df.to_csv(save_path / 'model_comparison_metrics.csv')
        print(f"\nModel comparison metrics saved to {save_path / 'model_comparison_metrics.csv'}")


def generate_evaluation_report(reg_model, clf_model, data_dict, results_dir='results'):
    """
    Generate comprehensive evaluation report for both models.
    
    Parameters:
    -----------
    reg_model : xgb.XGBRegressor
        Trained regression model
    clf_model : xgb.XGBClassifier
        Trained classification model
    data_dict : dict
        Dictionary containing test data for both models
    results_dir : str
        Directory to save results
    """
    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    reg_results = evaluate_regression_model(
        reg_model,
        data_dict['regression']['X_test'],
        data_dict['regression']['y_test'],
        save_path=results_path / 'regression'
    )
    
    clf_results = evaluate_classification_model(
        clf_model,
        data_dict['classification']['X_test'],
        data_dict['classification']['y_test'],
        save_path=results_path / 'classification'
    )
    
    compare_models(reg_results, clf_results, save_path=results_path)
    
    summary = {
        'Regression': {
            'RMSE': float(reg_results['RMSE']),
            'MAE': float(reg_results['MAE']),
            'R²': float(reg_results['R²'])
        },
        'Classification': {
            'Accuracy': float(clf_results['Accuracy']),
            'Precision': float(clf_results['Precision']),
            'Recall': float(clf_results['Recall']),
            'F1': float(clf_results['F1']),
            'ROC-AUC': float(clf_results['ROC-AUC'])
        }
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_path / 'evaluation_summary.csv')
    
    print(f"\n✓ Evaluation report complete! All results saved to {results_path}")
    print("\nSummary:")
    print(summary_df.to_string())

