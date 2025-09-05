import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering for chronic care risk prediction"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def create_features(self, df):
        """Create engineered features from raw data"""

        df_processed = df.copy()

        # Create age bands
        df_processed['age_band'] = pd.cut(
            df_processed['age'], 
            bins=[0, 50, 65, 75, 100], 
            labels=['<50', '50-64', '65-74', '75+']
        )

        # BMI categories
        df_processed['bmi_category'] = pd.cut(
            df_processed['bmi'],
            bins=[0, 18.5, 25, 30, 50],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

        # HbA1c control categories
        df_processed['hba1c_control'] = pd.cut(
            df_processed['hba1c_last'],
            bins=[0, 7, 9, 15],
            labels=['Good', 'Fair', 'Poor']
        )

        # Adherence categories
        df_processed['adherence_category'] = pd.cut(
            df_processed['adherence_mean'],
            bins=[0, 0.8, 0.9, 1.0],
            labels=['Poor', 'Fair', 'Good']
        )

        # Risk flags
        df_processed['hba1c_high'] = (df_processed['hba1c_last'] > 9).astype(int)
        df_processed['weight_gain_rapid'] = (df_processed['weight_trend_30d'] > 1.5).astype(int)
        df_processed['adherence_low'] = (df_processed['adherence_mean'] < 0.8).astype(int)
        df_processed['bnp_elevated'] = (df_processed['bnp_last'] > 400).astype(int)
        df_processed['egfr_declining'] = (df_processed['egfr_trend_90d'] < -5).astype(int)
        df_processed['bp_high'] = (df_processed['sbp_last'] > 140).astype(int)
        df_processed['care_gap'] = (df_processed['days_since_last_lab'] > 90).astype(int)

        # Interaction features
        df_processed['age_adherence_interaction'] = (
            df_processed['age'] * (1 - df_processed['adherence_mean'])
        )

        df_processed['diabetes_control_score'] = (
            df_processed['hba1c_last'] * (1 - df_processed['adherence_mean'])
        )

        # Risk score components
        df_processed['metabolic_risk'] = (
            df_processed['hba1c_high'] + 
            df_processed['bp_high'] + 
            (df_processed['bmi'] > 30).astype(int)
        )

        df_processed['cardiac_risk'] = (
            df_processed['bnp_elevated'] + 
            df_processed['weight_gain_rapid'] + 
            df_processed['bp_high']
        )

        return df_processed

    def prepare_model_features(self, df):
        """Prepare features for modeling"""

        # Select core numerical features
        numerical_features = [
            'age', 'hba1c_last', 'weight_trend_30d', 'adherence_mean',
            'bnp_last', 'egfr_trend_90d', 'sbp_last', 'bmi', 'days_since_last_lab'
        ]

        # Select binary flags
        flag_features = [
            'smoker', 'hba1c_high', 'weight_gain_rapid', 'adherence_low',
            'bnp_elevated', 'egfr_declining', 'bp_high', 'care_gap'
        ]

        # Select interaction features
        interaction_features = [
            'age_adherence_interaction', 'diabetes_control_score',
            'metabolic_risk', 'cardiac_risk'
        ]

        # Encode categorical features
        categorical_features = ['sex', 'condition_primary']
        df_encoded = df.copy()

        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df[col])

        # Combine all features
        all_features = (
            numerical_features + 
            flag_features + 
            interaction_features +
            [col + '_encoded' for col in categorical_features]
        )

        self.feature_names = all_features

        # Create feature matrix
        X = df_encoded[all_features]

        # Handle missing values
        X = X.fillna(X.median())

        return X

    def get_feature_descriptions(self):
        """Get human-readable feature descriptions"""

        descriptions = {
            'age': 'Patient Age (years)',
            'hba1c_last': 'HbA1c Last (%)',
            'weight_trend_30d': 'Weight Trend (kg/30d)',
            'adherence_mean': 'Medication Adherence (0-1)',
            'bnp_last': 'BNP Last (pg/mL)',
            'egfr_trend_90d': 'eGFR Trend (mL/min/1.73m²/90d)',
            'sbp_last': 'Systolic BP Last (mmHg)',
            'bmi': 'BMI (kg/m²)',
            'days_since_last_lab': 'Days Since Last Lab',
            'smoker': 'Smoking Status',
            'hba1c_high': 'HbA1c >9%',
            'weight_gain_rapid': 'Rapid Weight Gain',
            'adherence_low': 'Poor Adherence (<80%)',
            'bnp_elevated': 'BNP Elevated (>400)',
            'egfr_declining': 'eGFR Declining (<-5)',
            'bp_high': 'High Blood Pressure (>140)',
            'care_gap': 'Care Gap (>90 days)',
            'age_adherence_interaction': 'Age × Non-adherence',
            'diabetes_control_score': 'Diabetes Control Score',
            'metabolic_risk': 'Metabolic Risk Score',
            'cardiac_risk': 'Cardiac Risk Score',
            'sex_encoded': 'Sex (encoded)',
            'condition_primary_encoded': 'Primary Condition (encoded)'
        }

        return descriptions

def create_feature_summary(df):
    """Create feature summary for documentation"""

    summary = {
        'Total Features': len(df.columns),
        'Numerical Features': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical Features': len(df.select_dtypes(include=['object']).columns),
        'Missing Values': df.isnull().sum().sum(),
        'Data Shape': df.shape
    }

    return summary

if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")

    # Load data
    df = pd.read_csv('../data/processed/training_table.csv')

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Create features
    df_features = fe.create_features(df)
    X = fe.prepare_model_features(df_features)

    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_features.columns)}")
    print(f"Model features: {len(X.columns)}")

    print("\nFeature engineering completed successfully!")
