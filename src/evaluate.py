import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, calibration_curve
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from features import FeatureEngineer
from utils import plot_model_evaluation, generate_model_report

class ModelEvaluator:
    """Comprehensive model evaluation for CareGuard AI"""

    def __init__(self, model_path):
        """Initialize evaluator with trained model"""

        self.model_bundle = joblib.load(model_path)
        self.model = self.model_bundle['calibrated_model']
        self.scaler = self.model_bundle['scaler']
        self.feature_engineer = self.model_bundle['feature_engineer']
        self.threshold = self.model_bundle['threshold']

        self.evaluation_results = {}

    def prepare_test_data(self, test_data_path):
        """Prepare test data for evaluation"""

        df = pd.read_csv(test_data_path)

        # Engineer features
        df_features = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_model_features(df_features)
        y = df['label_90d'].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled, y, df_features

    def evaluate_performance(self, X_test, y_test):
        """Evaluate model performance"""

        print("Evaluating model performance...")

        # Generate predictions
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        # Basic metrics
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        # Confusion matrix and derived metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        self.evaluation_results = {
            'basic_metrics': {
                'AUROC': auroc,
                'AUPRC': auprc,
                'Brier_Score': brier,
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Precision': precision,
                'NPV': npv,
                'F1_Score': f1
            },
            'confusion_matrix': {
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp)
            },
            'classification_report': class_report,
            'threshold': self.threshold,
            'test_set_size': len(y_test),
            'positive_rate': float(y_test.mean())
        }

        return self.evaluation_results

    def evaluate_calibration(self, X_test, y_test, n_bins=10):
        """Evaluate model calibration"""

        print("Evaluating model calibration...")

        y_prob = self.model.predict_proba(X_test)[:, 1]

        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

            # Calculate calibration metrics
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            max_calibration_error = np.max(np.abs(prob_true - prob_pred))

            calibration_results = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist(),
                'mean_calibration_error': calibration_error,
                'max_calibration_error': max_calibration_error,
                'n_bins': n_bins
            }

            self.evaluation_results['calibration'] = calibration_results

        except Exception as e:
            print(f"Calibration evaluation failed: {e}")
            self.evaluation_results['calibration'] = {'error': str(e)}

        return self.evaluation_results.get('calibration', {})

    def evaluate_risk_bands(self, X_test, y_test):
        """Evaluate performance by risk bands"""

        print("Evaluating risk band performance...")

        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Define risk bands
        risk_bands = pd.cut(
            y_prob, 
            bins=[-0.01, 0.1, 0.25, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        band_analysis = {}

        for band in ['Low', 'Medium', 'High']:
            mask = (risk_bands == band)

            if mask.sum() > 0:
                band_y_true = y_test[mask]
                band_y_prob = y_prob[mask]
                band_y_pred = (band_y_prob >= self.threshold).astype(int)

                band_metrics = {
                    'count': int(mask.sum()),
                    'percentage': float(mask.sum() / len(y_test) * 100),
                    'positive_rate': float(band_y_true.mean()),
                    'mean_probability': float(band_y_prob.mean()),
                    'std_probability': float(band_y_prob.std()),
                    'min_probability': float(band_y_prob.min()),
                    'max_probability': float(band_y_prob.max())
                }

                # Band-specific performance metrics
                if len(np.unique(band_y_true)) > 1:  # Both classes present
                    try:
                        band_auroc = roc_auc_score(band_y_true, band_y_prob)
                        band_metrics['AUROC'] = band_auroc
                    except:
                        band_metrics['AUROC'] = None

                band_analysis[band] = band_metrics

        self.evaluation_results['risk_bands'] = band_analysis
        return band_analysis

    def evaluate_subgroups(self, X_test, y_test, df_features):
        """Evaluate performance across demographic subgroups"""

        print("Evaluating subgroup performance...")

        y_prob = self.model.predict_proba(X_test)[:, 1]

        subgroup_analysis = {}

        # Evaluate by sex
        if 'sex' in df_features.columns:
            for sex in df_features['sex'].unique():
                mask = (df_features['sex'] == sex)
                if mask.sum() > 10:  # Minimum sample size
                    subgroup_y_true = y_test[mask]
                    subgroup_y_prob = y_prob[mask]

                    if len(np.unique(subgroup_y_true)) > 1:
                        try:
                            auroc = roc_auc_score(subgroup_y_true, subgroup_y_prob)
                            auprc = average_precision_score(subgroup_y_true, subgroup_y_prob)

                            subgroup_analysis[f'sex_{sex}'] = {
                                'count': int(mask.sum()),
                                'positive_rate': float(subgroup_y_true.mean()),
                                'AUROC': auroc,
                                'AUPRC': auprc
                            }
                        except:
                            pass

        # Evaluate by age groups
        if 'age' in df_features.columns:
            age_groups = pd.cut(df_features['age'], bins=[0, 50, 65, 75, 100], labels=['<50', '50-64', '65-74', '75+'])

            for age_group in age_groups.unique():
                if pd.isna(age_group):
                    continue

                mask = (age_groups == age_group)
                if mask.sum() > 10:
                    subgroup_y_true = y_test[mask]
                    subgroup_y_prob = y_prob[mask]

                    if len(np.unique(subgroup_y_true)) > 1:
                        try:
                            auroc = roc_auc_score(subgroup_y_true, subgroup_y_prob)
                            auprc = average_precision_score(subgroup_y_true, subgroup_y_prob)

                            subgroup_analysis[f'age_{age_group}'] = {
                                'count': int(mask.sum()),
                                'positive_rate': float(subgroup_y_true.mean()),
                                'AUROC': auroc,
                                'AUPRC': auprc
                            }
                        except:
                            pass

        # Evaluate by primary condition
        if 'condition_primary' in df_features.columns:
            for condition in df_features['condition_primary'].unique():
                mask = (df_features['condition_primary'] == condition)
                if mask.sum() > 10:
                    subgroup_y_true = y_test[mask]
                    subgroup_y_prob = y_prob[mask]

                    if len(np.unique(subgroup_y_true)) > 1:
                        try:
                            auroc = roc_auc_score(subgroup_y_true, subgroup_y_prob)
                            auprc = average_precision_score(subgroup_y_true, subgroup_y_prob)

                            subgroup_analysis[f'condition_{condition}'] = {
                                'count': int(mask.sum()),
                                'positive_rate': float(subgroup_y_true.mean()),
                                'AUROC': auroc,
                                'AUPRC': auprc
                            }
                        except:
                            pass

        self.evaluation_results['subgroups'] = subgroup_analysis
        return subgroup_analysis

    def evaluate_clinical_utility(self, X_test, y_test):
        """Evaluate clinical utility metrics"""

        print("Evaluating clinical utility...")

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        # Calculate clinical utility metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        total_patients = len(y_test)

        clinical_metrics = {
            'patients_flagged': int(y_pred.sum()),
            'patients_flagged_pct': float(y_pred.sum() / total_patients * 100),
            'true_positives_found': int(tp),
            'false_positives': int(fp),
            'missed_cases': int(fn),
            'number_needed_to_screen': int(y_pred.sum() / tp) if tp > 0 else float('inf'),
            'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        }

        # Alert fatigue considerations
        clinical_metrics['alert_rate'] = clinical_metrics['patients_flagged_pct']
        clinical_metrics['precision_for_alerts'] = clinical_metrics['positive_predictive_value']

        self.evaluation_results['clinical_utility'] = clinical_metrics
        return clinical_metrics

    def generate_evaluation_plots(self, X_test, y_test, save_dir='../models/evaluation_plots'):
        """Generate evaluation plots"""

        import os
        os.makedirs(save_dir, exist_ok=True)

        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Comprehensive evaluation plot
        fig = plot_model_evaluation(y_test, y_prob, self.threshold, 
                                  save_path=f'{save_dir}/model_evaluation.png')

        # Feature importance plot
        try:
            if 'primary_model' in self.model_bundle:
                importances = self.model_bundle['primary_model'].feature_importances_
                features = self.model_bundle['features']

                # Sort features by importance
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=True).tail(15)

                plt.figure(figsize=(10, 8))
                plt.barh(importance_df['feature'], importance_df['importance'])
                plt.xlabel('Feature Importance')
                plt.title('Top 15 Feature Importances (XGBoost)')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()

        except Exception as e:
            print(f"Could not generate feature importance plot: {e}")

        return save_dir

    def generate_comprehensive_report(self, X_test, y_test, df_features, save_path='../models/evaluation_report.json'):
        """Generate comprehensive evaluation report"""

        print("Generating comprehensive evaluation report...")

        # Run all evaluations
        self.evaluate_performance(X_test, y_test)
        self.evaluate_calibration(X_test, y_test)
        self.evaluate_risk_bands(X_test, y_test)
        self.evaluate_subgroups(X_test, y_test, df_features)
        self.evaluate_clinical_utility(X_test, y_test)

        # Add metadata
        self.evaluation_results['metadata'] = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_threshold': self.threshold,
            'test_set_size': len(y_test),
            'evaluator_version': '1.0.0'
        }

        # Save report
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)

        print(f"Comprehensive evaluation report saved to {save_path}")

        return self.evaluation_results

    def print_summary(self):
        """Print evaluation summary"""

        if not self.evaluation_results:
            print("No evaluation results available. Run evaluation first.")
            return

        print("\n" + "="*50)
        print("CAREGUARD AI MODEL EVALUATION SUMMARY")
        print("="*50)

        # Basic metrics
        if 'basic_metrics' in self.evaluation_results:
            metrics = self.evaluation_results['basic_metrics']
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   AUROC: {metrics['AUROC']:.3f}")
            print(f"   AUPRC: {metrics['AUPRC']:.3f}")
            print(f"   Sensitivity: {metrics['Sensitivity']:.3f}")
            print(f"   Specificity: {metrics['Specificity']:.3f}")
            print(f"   Precision: {metrics['Precision']:.3f}")
            print(f"   F1 Score: {metrics['F1_Score']:.3f}")
            print(f"   Brier Score: {metrics['Brier_Score']:.3f}")

        # Clinical utility
        if 'clinical_utility' in self.evaluation_results:
            clinical = self.evaluation_results['clinical_utility']
            print(f"\n🏥 CLINICAL UTILITY:")
            print(f"   Patients Flagged: {clinical['patients_flagged']} ({clinical['patients_flagged_pct']:.1f}%)")
            print(f"   True Positives Found: {clinical['true_positives_found']}")
            print(f"   Number Needed to Screen: {clinical['number_needed_to_screen']}")
            print(f"   Alert Precision: {clinical['precision_for_alerts']:.3f}")

        # Risk bands
        if 'risk_bands' in self.evaluation_results:
            print(f"\n📈 RISK BAND ANALYSIS:")
            for band, metrics in self.evaluation_results['risk_bands'].items():
                print(f"   {band} Risk: {metrics['count']} patients ({metrics['percentage']:.1f}%), {metrics['positive_rate']:.1%} deterioration rate")

        print("\n" + "="*50)

def main():
    """Main evaluation pipeline"""

    print("=== CareGuard AI Model Evaluation ===\n")

    # Initialize evaluator
    evaluator = ModelEvaluator('../models/model.pkl')

    # Prepare test data
    X_test, y_test, df_features = evaluator.prepare_test_data('../data/processed/training_table.csv')

    # Use a subset for evaluation (last 25% as test set)
    test_size = int(0.25 * len(X_test))
    X_test = X_test[-test_size:]
    y_test = y_test[-test_size:]
    df_features = df_features.tail(test_size).reset_index(drop=True)

    print(f"Evaluating on test set of {len(y_test)} patients")

    # Generate comprehensive report
    results = evaluator.generate_comprehensive_report(X_test, y_test, df_features)

    # Generate plots
    evaluator.generate_evaluation_plots(X_test, y_test)

    # Print summary
    evaluator.print_summary()

    print("\n=== Evaluation Complete! ===")

    return evaluator

if __name__ == "__main__":
    evaluator = main()
