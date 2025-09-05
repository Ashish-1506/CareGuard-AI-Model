import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, 
    brier_score_loss, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from features import FeatureEngineer

class ChronicRiskPredictor:
    """Main class for chronic disease risk prediction"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.baseline_model = None
        self.primary_model = None
        self.calibrated_model = None
        self.threshold = 0.25
        self.metrics = {}

    def load_and_prepare_data(self, data_path):
        """Load and prepare data for training"""

        print("Loading data...")
        df = pd.read_csv(data_path)

        print("Engineering features...")
        df_features = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_model_features(df_features)
        y = df['label_90d'].values

        print(f"Data shape: {X.shape}")
        print(f"Positive class rate: {y.mean():.3f}")

        return X, y, df_features

    def split_data(self, X, y, test_size=0.25, random_state=42):
        """Split data into train and test sets"""

        return train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=random_state
        )

    def train_baseline_model(self, X_train, y_train):
        """Train baseline logistic regression model"""

        print("\nTraining baseline logistic regression model...")

        self.baseline_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )

        self.baseline_model.fit(X_train, y_train)

        return self.baseline_model

    def train_primary_model(self, X_train, y_train, X_val, y_val):
        print("Training primary XGBoost model...")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.primary_model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )

        from xgboost.callback import EarlyStopping

        self.primary_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        return self.primary_model


    def calibrate_model(self, X_train, y_train):
        """Apply calibration to the model"""

        print("Calibrating model...")

        self.calibrated_model = CalibratedClassifierCV(
            self.primary_model, 
            method='isotonic', 
            cv=3
        )

        self.calibrated_model.fit(X_train, y_train)

        return self.calibrated_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and compute metrics"""

        print("\nEvaluating models...")

        models = {
            'baseline': self.baseline_model,
            'xgboost': self.primary_model,
            'calibrated': self.calibrated_model
        }

        results = {}

        for name, model in models.items():
            if model is None:
                continue

            # Predictions
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            # Metrics
            metrics = {
                'AUROC': roc_auc_score(y_test, probs),
                'AUPRC': average_precision_score(y_test, probs),
                'Brier_Score': brier_score_loss(y_test, probs),
                'Threshold': self.threshold
            }

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            metrics['Confusion_Matrix'] = cm.tolist()

            # Additional metrics
            tn, fp, fn, tp = cm.ravel()
            metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0

            results[name] = metrics

        self.metrics = results
        return results

    def optimize_threshold(self, X_val, y_val, model_type='calibrated'):
        """Optimize threshold using validation data"""

        model = getattr(self, f'{model_type}_model')
        if model is None:
            print(f"No {model_type} model found")
            return self.threshold

        probs = model.predict_proba(X_val)[:, 1]

        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Also consider F1 score optimization
        precision, recall, pr_thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        f1_optimal_threshold = pr_thresholds[f1_optimal_idx]

        # Use average of both methods
        self.threshold = (optimal_threshold + f1_optimal_threshold) / 2

        print(f"Optimized threshold: {self.threshold:.3f}")
        print(f"  - Youden's J: {optimal_threshold:.3f}")
        print(f"  - F1 optimal: {f1_optimal_threshold:.3f}")

        return self.threshold

    def save_model(self, save_path):
        """Save the trained model and artifacts"""

        bundle = {
            'calibrated_model': self.calibrated_model,
            'primary_model': self.primary_model,
            'baseline_model': self.baseline_model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'features': self.feature_engineer.feature_names,
            'threshold': self.threshold,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(bundle, save_path)
        print(f"Model saved to {save_path}")

    def get_feature_importance(self, top_n=15):
        """Get feature importance from XGBoost model"""

        if self.primary_model is None:
            return None

        importance = self.primary_model.feature_importances_
        feature_names = self.feature_engineer.feature_names

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

def main():
    print("=== CareGuard AI Training Pipeline ===\n")
    predictor = ChronicRiskPredictor()
    # Load and prepare data
    X, y, df_features = predictor.load_and_prepare_data('../data/processed/training_table.csv')

    # Split data into big train and test set first
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)

    # Now, split X_train further into train and val
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=123
    )

    # Scale features
    print("Scaling features...")
    X_train_main_scaled = predictor.scaler.fit_transform(X_train_main)
    X_val_scaled = predictor.scaler.transform(X_val)
    X_test_scaled = predictor.scaler.transform(X_test)

    # Train models
    predictor.train_baseline_model(X_train_main_scaled, y_train_main)
    predictor.train_primary_model(X_train_main_scaled, y_train_main, X_val_scaled, y_val)  # pass val set

    # Calibrate model on full training set
    predictor.calibrate_model(X_train_main_scaled, y_train_main)

    # Evaluate models
    results = predictor.evaluate_models(X_test_scaled, y_test)

    # Print results
    print("\n=== Model Performance ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        for metric, value in metrics.items():
            if metric == 'Confusion_Matrix':
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value:.4f}")

    # Feature importance
    importance_df = predictor.get_feature_importance()
    print("\n=== Top Feature Importance ===")
    print(importance_df)

    # Save model
    predictor.save_model('../models/model.pkl')

    # Save metrics to JSON
    import json
    with open('../models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Training Complete! ===")
    return predictor


if __name__ == "__main__":
    predictor = main()
