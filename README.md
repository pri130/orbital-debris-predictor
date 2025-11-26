# Orbital Debris Collision Predictor

End-to-end pipeline to predict satellite-debris collision risk using public TLEs (Celestrak),
SGP4 propagation, KD-tree screening, analytic closest-approach features, and ML (RandomForest/XGBoost).

## Quick demo (Colab)
1. Open Colab and upload models or mount Google Drive where models are stored.
2. Open `notebooks/pipeline_demo.ipynb` or use this Colab link pattern:
   https://colab.research.google.com/github/<yourusername>/orbital-debris-predictor/blob/main/notebooks/pipeline_demo.ipynb

## Files
- `src/` : scripts (parse_tles.py, propagate.py, screening.py, features.py, train.py, predict.py, app_streamlit.py)
- `notebooks/` : pipeline demo notebook
- `data/` : sample_features.csv (small sample)
- `models/` : rf_model.joblib, xgb_model.json (optional to include; use Drive if large)

## Quick start (local)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run demo notebook (recommended in Colab)
# Or run scripts locally, for example:
python src/predict.py --features data/sample_features.csv --model models/rf_model.joblib
