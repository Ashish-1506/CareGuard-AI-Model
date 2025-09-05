import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve
import joblib
import json
from datetime import datetime
import os

def plot_model_evaluation(y_true, y_prob, threshold=0.25, save_path=None):
    """Plot comprehensive model evaluation metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = np.trapz(tpr, fpr)

    axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)

    axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', label='Random')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Calibration Plot
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        axes[1, 0].plot(prob_pred, prob_true, marker='o', label='Model')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title('Calibration Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    except:
        axes[1, 0].text(0.5, 0.5, 'Calibration plot unavailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    # Distribution of Predictions
    axes[1, 1].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Negative', density=True)
    axes[1, 1].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Positive', density=True)
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plot saved to {save_path}")

    plt.show()
    return fig

def generate_model_report(model_bundle, X_test, y_test, save_path=None):
    """Generate comprehensive model report"""

    model = model_bundle['calibrated_model']
    threshold = model_bundle['threshold']
    features = model_bundle['features']
    metrics = model_bundle.get('metrics', {})

    # Generate predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Create report
    report = {
        'Model Information': {
            'Model Type': 'XGBoost with Isotonic Calibration',
            'Number of Features': len(features),
            'Threshold': threshold,
            'Training Timestamp': model_bundle.get('timestamp', 'Unknown')
        },
        'Dataset Information': {
            'Test Set Size': len(y_test),
            'Positive Class Rate': float(y_test.mean()),
            'Class Distribution': {
                'Negative': int((y_test == 0).sum()),
                'Positive': int((y_test == 1).sum())
            }
        },
        'Performance Metrics': metrics,
        'Feature Importance': get_feature_importance_dict(model_bundle),
        'Risk Band Analysis': analyze_risk_bands(y_prob, y_test, threshold)
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Model report saved to {save_path}")

    return report

def get_feature_importance_dict(model_bundle):
    """Get feature importance as dictionary"""

    try:
        model = model_bundle['primary_model']  # Use XGBoost for importance
        features = model_bundle['features']

        importance = model.feature_importances_

        importance_dict = {}
        for feature, imp in zip(features, importance):
            importance_dict[feature] = float(imp)

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True))

        return sorted_importance

    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return {}

def analyze_risk_bands(y_prob, y_test, threshold):
    """Analyze performance by risk bands"""

    # Define risk bands
    risk_bands = pd.cut(y_prob, bins=[-0.01, 0.1, 0.25, 1.0], labels=['Low', 'Medium', 'High'])

    analysis = {}

    for band in ['Low', 'Medium', 'High']:
        mask = (risk_bands == band)
        if mask.sum() > 0:
            band_y_true = y_test[mask]
            band_y_prob = y_prob[mask]

            analysis[band] = {
                'Count': int(mask.sum()),
                'Percentage': float(mask.sum() / len(y_test) * 100),
                'Positive_Rate': float(band_y_true.mean()),
                'Mean_Probability': float(band_y_prob.mean()),
                'Min_Probability': float(band_y_prob.min()),
                'Max_Probability': float(band_y_prob.max())
            }

    return analysis

def create_synthetic_patient(base_patient=None, risk_level='medium'):
    """Create synthetic patient for testing"""

    np.random.seed(42)

    if base_patient is None:
        if risk_level == 'low':
            patient = {
                'age': np.random.randint(40, 55),
                'sex': np.random.choice(['M', 'F']),
                'condition_primary': 'Diabetes',
                'hba1c_last': np.random.uniform(6.0, 7.5),
                'weight_trend_30d': np.random.uniform(-0.5, 0.5),
                'adherence_mean': np.random.uniform(0.85, 0.98),
                'bnp_last': np.random.randint(50, 150),
                'egfr_trend_90d': np.random.uniform(-2, 2),
                'sbp_last': np.random.randint(110, 130),
                'bmi': np.random.uniform(22, 28),
                'days_since_last_lab': np.random.randint(15, 60),
                'smoker': 0
            }
        elif risk_level == 'high':
            patient = {
                'age': np.random.randint(65, 80),
                'sex': np.random.choice(['M', 'F']),
                'condition_primary': 'Multiple',
                'hba1c_last': np.random.uniform(9.5, 12.0),
                'weight_trend_30d': np.random.uniform(1.5, 4.0),
                'adherence_mean': np.random.uniform(0.4, 0.7),
                'bnp_last': np.random.randint(400, 800),
                'egfr_trend_90d': np.random.uniform(-15, -5),
                'sbp_last': np.random.randint(150, 180),
                'bmi': np.random.uniform(32, 40),
                'days_since_last_lab': np.random.randint(120, 300),
                'smoker': 1
            }
        else:  # medium risk
            patient = {
                'age': np.random.randint(55, 70),
                'sex': np.random.choice(['M', 'F']),
                'condition_primary': np.random.choice(['Diabetes', 'Heart Failure', 'Hypertension']),
                'hba1c_last': np.random.uniform(7.5, 9.0),
                'weight_trend_30d': np.random.uniform(0.5, 1.5),
                'adherence_mean': np.random.uniform(0.7, 0.85),
                'bnp_last': np.random.randint(150, 400),
                'egfr_trend_90d': np.random.uniform(-8, -2),
                'sbp_last': np.random.randint(130, 150),
                'bmi': np.random.uniform(28, 35),
                'days_since_last_lab': np.random.randint(60, 120),
                'smoker': np.random.choice([0, 1])
            }
    else:
        patient = base_patient.copy()

    # Add additional fields
    patient['patient_id'] = f"TEST_{np.random.randint(90000, 99999)}"
    patient['patient_name'] = f"Test Patient {patient['patient_id'][-3:]}"
    patient['last_updated'] = datetime.now()

    return pd.DataFrame([patient])

def validate_model_inputs(patient_data, feature_names):
    """Validate model inputs"""

    errors = []

    # Check required features
    missing_features = set(feature_names) - set(patient_data.columns)
    if missing_features:
        errors.append(f"Missing features: {missing_features}")

    # Check data types and ranges
    numeric_ranges = {
        'age': (18, 120),
        'hba1c_last': (4.0, 15.0),
        'weight_trend_30d': (-10.0, 10.0),
        'adherence_mean': (0.0, 1.0),
        'bnp_last': (0, 5000),
        'egfr_trend_90d': (-50, 50),
        'sbp_last': (70, 250),
        'bmi': (15, 60),
        'days_since_last_lab': (0, 730)
    }

    for feature, (min_val, max_val) in numeric_ranges.items():
        if feature in patient_data.columns:
            values = patient_data[feature]
            if values.isna().any():
                errors.append(f"{feature} contains missing values")
            elif (values < min_val).any() or (values > max_val).any():
                errors.append(f"{feature} values outside valid range ({min_val}-{max_val})")

    return errors

def format_clinical_output(prediction_result, explanation_result):
    """Format results for clinical output"""

    output = {
        'timestamp': datetime.now().isoformat(),
        'risk_assessment': {
            'probability': f"{prediction_result['risk_probability']:.1%}",
            'band': prediction_result['risk_band'],
            'confidence': 'High' if abs(prediction_result['risk_probability'] - 0.5) > 0.3 else 'Medium'
        },
        'clinical_summary': explanation_result['clinical_summary'],
        'key_drivers': [
            {
                'factor': driver['description'],
                'impact': driver['impact'],
                'strength': 'Strong' if abs(driver['shap_value']) > 0.1 else 'Moderate'
            }
            for driver in explanation_result['top_drivers'][:3]
        ],
        'recommendations': explanation_result['recommendations'],
        'next_steps': {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
    }

    # Categorize recommendations
    for rec in explanation_result['recommendations']:
        if any(keyword in rec.lower() for keyword in ['urgent', 'immediate', 'today', 'now']):
            output['next_steps']['immediate'].append(rec)
        elif any(keyword in rec.lower() for keyword in ['30', 'days', 'follow-up', 'recheck']):
            output['next_steps']['short_term'].append(rec)
        else:
            output['next_steps']['long_term'].append(rec)

    return output

def save_results(results, filename, format='json'):
    """Save results in specified format"""

    if format.lower() == 'json':
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    elif format.lower() == 'csv' and isinstance(results, dict):
        # Flatten dictionary for CSV
        flattened = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = subvalue
            else:
                flattened[key] = value

        pd.DataFrame([flattened]).to_csv(filename, index=False)

    else:
        raise ValueError("Unsupported format. Use 'json' or 'csv'.")

    print(f"Results saved to {filename}")

def setup_logging():
    """Setup logging configuration"""

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/careguard.log'),
            logging.StreamHandler()
        ]
    )

    # Create logs directory if it doesn't exist
    os.makedirs('../logs', exist_ok=True)

    return logging.getLogger('CareGuardAI')

if __name__ == "__main__":
    print("CareGuard AI Utilities Module")
    print("Functions available:")
    print("- plot_model_evaluation()")
    print("- generate_model_report()")
    print("- create_synthetic_patient()")
    print("- validate_model_inputs()")
    print("- format_clinical_output()")
    print("- save_results()")
    print("- setup_logging()")
