import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    """Generate synthetic chronic care patient data for risk prediction"""

    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    # Number of patients
    N = 1500

    # Generate patient demographics
    patient_ids = np.arange(10000, 10000+N)
    ages = rng.integers(40, 85, size=N)
    sex = rng.choice(["F", "M"], size=N, p=[0.52, 0.48])

    # Chronic conditions distribution
    conditions = rng.choice(
        ["Diabetes", "Heart Failure", "Hypertension", "Multiple"], 
        size=N, 
        p=[0.35, 0.25, 0.25, 0.15]
    )

    # Baseline risk factors (age and sex influence)
    base_risk = (ages - 40) / 60 + (sex == "M") * 0.05

    # Generate clinical features with realistic distributions

    # HbA1c (normal: <7%, pre-diabetic: 7-9%, diabetic: >9%)
    hba1c_base = np.where(
        np.isin(conditions, ['Diabetes', 'Multiple']), 8.2, 6.8
    )
    hba1c_last = np.clip(
        rng.normal(hba1c_base + base_risk * 1.5, 1.2, N), 
        5.2, 13.0
    )

    # Weight trend (kg/30 days) - heart failure patients tend to gain weight
    weight_trend_base = np.where(
        np.isin(conditions, ['Heart Failure', 'Multiple']), 0.8, 0.1
    )
    weight_trend_30d = rng.normal(
        weight_trend_base + base_risk * 1.2, 0.8, N
    )

    # Medication adherence (0-1 scale)
    adherence_base = np.where(conditions == 'Multiple', 0.75, 0.85)
    adherence_mean = np.clip(
        rng.normal(adherence_base - base_risk * 0.15, 0.12, N), 
        0.3, 1.0
    )

    # BNP (B-type Natriuretic Peptide) - heart failure marker
    bnp_base = np.where(
        np.isin(conditions, ['Heart Failure', 'Multiple']), 350, 80
    )
    bnp_last = np.clip(
        rng.normal(bnp_base + base_risk * 200, 120, N), 
        20, 2500
    )

    # eGFR trend (kidney function decline, mL/min/1.73m²/90d)
    egfr_trend_base = np.where(conditions == 'Multiple', -4, -1)
    egfr_trend_90d = rng.normal(
        egfr_trend_base - base_risk * 3, 4, N
    )

    # Systolic Blood Pressure (last reading)
    sbp_base = np.where(
        np.isin(conditions, ['Hypertension', 'Multiple']), 145, 125
    )
    sbp_last = np.clip(
        rng.normal(sbp_base + base_risk * 15, 18, N), 
        90, 200
    )

    # Additional risk factors
    bmi = np.clip(rng.normal(28 + base_risk * 5, 6, N), 18, 45)

    # Days since last lab/vitals (care engagement proxy)
    days_since_last_lab = np.clip(
        rng.exponential(45 + base_risk * 30, N), 1, 365
    ).astype(int)

    # Smoking status
    smoker = rng.choice([0, 1], size=N, p=[0.78, 0.22])

    # Generate outcome label (deterioration in next 90 days)
    # Logistic function combining multiple risk factors
    logit = (
        -2.5 +  # Base intercept
        0.45 * (hba1c_last - 7) +  # HbA1c effect
        0.6 * (weight_trend_30d > 1.5) +  # Rapid weight gain
        0.8 * (adherence_mean < 0.8) +  # Poor adherence
        0.002 * (bnp_last - 100) +  # BNP elevation
        -0.03 * egfr_trend_90d +  # Kidney function decline
        0.02 * (sbp_last - 130) +  # Blood pressure
        0.03 * (bmi - 25) +  # BMI effect
        0.001 * days_since_last_lab +  # Poor engagement
        0.3 * smoker +  # Smoking
        0.02 * (ages - 60)  # Age effect
    )

    # Convert to probability and generate labels
    prob_deterioration = 1 / (1 + np.exp(-logit))
    prob_deterioration = np.clip(prob_deterioration, 0.02, 0.6)
    label_90d = rng.binomial(1, prob_deterioration)

    # Create DataFrame
    patients_df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'sex': sex,
        'condition_primary': conditions,
        'hba1c_last': np.round(hba1c_last, 1),
        'weight_trend_30d': np.round(weight_trend_30d, 2),
        'adherence_mean': np.round(adherence_mean, 3),
        'bnp_last': np.round(bnp_last, 0).astype(int),
        'egfr_trend_90d': np.round(egfr_trend_90d, 1),
        'sbp_last': np.round(sbp_last, 0).astype(int),
        'bmi': np.round(bmi, 1),
        'days_since_last_lab': days_since_last_lab,
        'smoker': smoker,
        'label_90d': label_90d,
        'true_probability': np.round(prob_deterioration, 4)
    })

    # Add patient names for demo purposes
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'David', 'Elizabeth'] * 150
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'] * 150

    patients_df['patient_name'] = [
        f"{rng.choice(first_names)} {rng.choice(last_names)}" 
        for _ in range(N)
    ]

    # Add last updated timestamp
    base_date = datetime.now() - timedelta(days=7)
    patients_df['last_updated'] = [
    base_date + timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
    for _ in range(N)
    ]


    return patients_df

if __name__ == "__main__":
    # Generate data
    print("Generating synthetic chronic care patient data...")
    df = generate_synthetic_data()

    # Save to CSV
    df.to_csv('../data/processed/training_table.csv', index=False)

    print(f"Generated data for {len(df)} patients")
    print(f"Deterioration rate: {df['label_90d'].mean():.2%}")
    print("\nData summary:")
    print(df.describe())

    print("\nCondition distribution:")
    print(df['condition_primary'].value_counts())

    print("\nData saved to data/processed/training_table.csv")
