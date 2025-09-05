import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """SHAP-based explainability for chronic risk prediction models"""

    def __init__(self, model_path):
        """Initialize explainer with trained model"""

        print("Loading model bundle...")
        self.bundle = joblib.load(model_path)

        self.model = self.bundle['primary_model']
        self.scaler = self.bundle['scaler']
        self.feature_engineer = self.bundle['feature_engineer']
        self.feature_names = self.bundle['features']
        self.threshold = self.bundle['threshold']

        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        self.explainer = None
        self.shap_values = None

    def prepare_data(self, df):
        """Prepare data for explanation"""

        # Engineer features
        df_features = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_model_features(df_features)
        X_scaled = self.scaler.transform(X)

        return X_scaled, X, df_features

    def fit_explainer(self, X_scaled, max_samples=100):
        """Fit SHAP explainer on sample data"""

        # Use a sample for explainer initialization (for performance)
        sample_size = min(max_samples, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        # Initialize explainer
        self.explainer = shap.Explainer(self.model, X_sample)

        print(f"SHAP explainer fitted on {sample_size} samples")

    def explain_cohort(self, X_scaled, max_samples=200):
        """Generate SHAP values for cohort"""

        if self.explainer is None:
            self.fit_explainer(X_scaled)

        # Calculate SHAP values for sample
        sample_size = min(max_samples, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        print("Calculating SHAP values...")
        self.shap_values = self.explainer(X_sample)

        return self.shap_values, sample_indices

    def explain_patient(self, patient_data, patient_id=None):
        """Generate explanation for single patient"""

        if isinstance(patient_data, pd.DataFrame):
            X_scaled, X_raw, df_features = self.prepare_data(patient_data)
        else:
            X_scaled = patient_data.reshape(1, -1)
            X_raw = None
            df_features = None

        if self.explainer is None:
            # Create a dummy background for single prediction
            background = np.zeros((10, X_scaled.shape[1]))
            self.explainer = shap.Explainer(self.model, background)

        # Get SHAP values
        shap_vals = self.explainer(X_scaled)

        # Get prediction
        prob = self.model.predict_proba(X_scaled)[0, 1]
        risk_band = self.get_risk_band(prob)

        # Format explanation
        explanation = self.format_patient_explanation(
            shap_vals[0], X_raw, prob, risk_band, patient_id
        )

        return explanation

    def format_patient_explanation(self, shap_values, X_raw, prob, risk_band, patient_id):
        """Format patient explanation in clinician-friendly language"""

        # Get feature descriptions
        descriptions = self.feature_engineer.get_feature_descriptions()

        # Sort features by absolute SHAP value
        feature_impact = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values.values)):

            feature_desc = descriptions.get(feature, feature)
            raw_value = X_raw.iloc[0, i] if X_raw is not None else "N/A"

            feature_impact.append({
                'feature': feature,
                'description': feature_desc,
                'shap_value': shap_val,
                'raw_value': raw_value,
                'impact': 'increases' if shap_val > 0 else 'decreases',
                'magnitude': abs(shap_val)
            })

        # Sort by magnitude
        feature_impact.sort(key=lambda x: x['magnitude'], reverse=True)

        # Create explanation text
        explanation = {
            'patient_id': patient_id,
            'risk_probability': prob,
            'risk_band': risk_band,
            'top_drivers': feature_impact[:5],
            'clinical_summary': self.generate_clinical_summary(feature_impact[:5]),
            'recommendations': self.generate_recommendations(feature_impact[:5])
        }

        return explanation

    def generate_clinical_summary(self, top_drivers):
        """Generate clinical summary from top drivers"""

        summary_parts = []

        for driver in top_drivers[:3]:  # Top 3 drivers
            feature = driver['feature']
            value = driver['raw_value']
            impact = driver['impact']

            if feature == 'hba1c_last' and driver['shap_value'] > 0:
                summary_parts.append(f"HbA1c {value:.1f}% indicates suboptimal diabetes control")
            elif feature == 'adherence_mean' and driver['shap_value'] > 0:
                summary_parts.append(f"Medication adherence {value:.0%} is below optimal")
            elif feature == 'weight_trend_30d' and driver['shap_value'] > 0:
                summary_parts.append(f"Weight trend +{value:.1f} kg/30d suggests fluid retention")
            elif feature == 'bnp_last' and driver['shap_value'] > 0:
                summary_parts.append(f"BNP {value:.0f} pg/mL indicates cardiac stress")
            elif feature == 'egfr_trend_90d' and driver['shap_value'] > 0:
                summary_parts.append(f"eGFR declining {value:.1f} mL/min/1.73m²/90d suggests kidney dysfunction")
            elif feature == 'sbp_last' and driver['shap_value'] > 0:
                summary_parts.append(f"Systolic BP {value:.0f} mmHg indicates hypertension")
            else:
                summary_parts.append(f"{driver['description']} {impact} risk")

        return '. '.join(summary_parts[:3]) + '.'

    def generate_recommendations(self, top_drivers):
        """Generate clinical recommendations based on drivers"""

        recommendations = []

        for driver in top_drivers:
            feature = driver['feature']
            value = driver['raw_value']

            if feature == 'hba1c_last' and value > 9:
                recommendations.append("Consider therapy intensification; schedule endocrinology consult; recheck HbA1c in 30-45 days")
            elif feature == 'adherence_mean' and value < 0.8:
                recommendations.append("Enroll in adherence support program; consider once-daily regimen; enable refill reminders")
            elif feature == 'weight_trend_30d' and value > 1.5:
                recommendations.append("Assess for fluid overload; adjust diuretics if indicated; schedule nurse call within 24-48 hours")
            elif feature == 'egfr_trend_90d' and value < -5:
                recommendations.append("Review nephrotoxic medications; order comprehensive metabolic panel; consider nephrology consult")
            elif feature == 'bnp_last' and value > 400:
                recommendations.append("Evaluate for heart failure decompensation; consider echocardiogram; optimize heart failure medications")
            elif feature == 'sbp_last' and value > 140:
                recommendations.append("Optimize antihypertensive therapy; lifestyle counseling; follow-up BP monitoring")
            elif feature == 'days_since_last_lab' and value > 90:
                recommendations.append("Schedule laboratory follow-up; ensure continuity of care; patient outreach")

        # Add default recommendation if none specific
        if not recommendations:
            recommendations.append("Continue current management; routine follow-up; reinforce adherence and lifestyle modifications")

        return recommendations[:3]  # Top 3 recommendations

    def get_risk_band(self, probability):
        """Convert probability to risk band"""

        if probability >= 0.25:
            return 'High'
        elif probability >= 0.10:
            return 'Medium'
        else:
            return 'Low'

    def plot_global_summary(self, save_path=None):
        """Plot global SHAP summary"""

        if self.shap_values is None:
            print("No SHAP values available. Run explain_cohort first.")
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.feature_names, 
            show=False,
            max_display=15
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_feature_importance(self, save_path=None):
        """Plot feature importance from SHAP values"""

        if self.shap_values is None:
            print("No SHAP values available. Run explain_cohort first.")
            return

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=15
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

def main():
    """Test explainability module"""

    print("=== Testing Model Explainability ===\n")

    # Load explainer
    explainer = ModelExplainer('../models/model.pkl')

    # Load test data
    df = pd.read_csv('../data/processed/training_table.csv').head(100)

    # Prepare data
    X_scaled, X_raw, df_features = explainer.prepare_data(df)

    # Generate cohort explanations
    shap_values, sample_indices = explainer.explain_cohort(X_scaled, max_samples=50)

    # Plot global summary
    print("Generating global SHAP summary...")
    explainer.plot_global_summary('../models/shap_summary.png')

    # Explain individual patient
    patient_sample = df.head(1)
    patient_explanation = explainer.explain_patient(patient_sample, patient_id="10000")

    print("\n=== Individual Patient Explanation ===")
    print(f"Patient ID: {patient_explanation['patient_id']}")
    print(f"Risk Probability: {patient_explanation['risk_probability']:.2%}")
    print(f"Risk Band: {patient_explanation['risk_band']}")
    print(f"\nClinical Summary: {patient_explanation['clinical_summary']}")
    print("\nTop Risk Drivers:")
    for driver in patient_explanation['top_drivers']:
        print(f"  - {driver['description']}: {driver['impact']} risk (SHAP: {driver['shap_value']:.3f})")
    print("\nRecommendations:")
    for rec in patient_explanation['recommendations']:
        print(f"  • {rec}")

    print("\n=== Explainability testing complete! ===")

    return explainer

if __name__ == "__main__":
    explainer = main()
