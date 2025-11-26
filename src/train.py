"""

Train baseline RandomForest and XGBoost on features.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import xgboost as xgb

def train_and_save(features_csv, rf_out="models/rf_model.joblib", xgb_out="models/xgb_model.json"):
    df = pd.read_csv(features_csv)
    X = df[["min_dist_m","rel_speed_m_s","closing_rate_m_s","time_to_CA_s"]]
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    classes = np.array([0,1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    cw = {0: weights[0], 1: weights[1]}
    rf = RandomForestClassifier(n_estimators=500, class_weight=cw, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_out)
    xgbm = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.7, colsample_bytree=0.7, scale_pos_weight=cw[1]/cw[0], use_label_encoder=False, eval_metric='logloss')
    xgbm.fit(X_train, y_train)
    xgbm.save_model(xgb_out)
    # evaluation
    rf_probs = rf.predict_proba(X_val)[:,1]
    xgb_probs = xgbm.predict_proba(X_val)[:,1]
    print("RF ROC-AUC:", roc_auc_score(y_val, rf_probs))
    print("RF PR-AUC :", average_precision_score(y_val, rf_probs))
    print("XGB ROC-AUC:", roc_auc_score(y_val, xgb_probs))
    print("XGB PR-AUC :", average_precision_score(y_val, xgb_probs))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/sample_features.csv")
    args = parser.parse_args()
    train_and_save(args.features)
