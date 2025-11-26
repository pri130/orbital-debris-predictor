"""

Load models and features.csv -> produce predictions.csv and alerts.csv
"""
import pandas as pd
import joblib
import xgboost as xgb

def predict_and_alert(features_csv, rf_model="models/rf_model.joblib", xgb_model="models/xgb_model.json", out_csv="data/predictions.csv", alert_csv="data/alerts.csv", alert_threshold=0.5):
    df = pd.read_csv(features_csv)
    X = df[["min_dist_m","rel_speed_m_s","closing_rate_m_s","time_to_CA_s"]].fillna(0.0)
    preds = {}
    if os.path.exists(rf_model):
        rf = joblib.load(rf_model)
        preds["rf"] = rf.predict_proba(X)[:,1]
    if os.path.exists(xgb_model):
        xgbm = xgb.XGBClassifier()
        xgbm.load_model(xgb_model)
        preds["xgb"] = xgbm.predict_proba(X)[:,1]
    if not preds:
        raise RuntimeError("No models found.")
    df["prob_ensemble"] = sum(preds.values()) / len(preds)
    df.to_csv(out_csv, index=False)
    alerts = df[df["prob_ensemble"] >= alert_threshold]
    alerts.to_csv(alert_csv, index=False)
    return df, alerts

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/sample_features.csv")
    args = parser.parse_args()
    df, alerts = predict_and_alert(args.features)
    print("Predictions saved. Alerts:", len(alerts))
