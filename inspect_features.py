import joblib

scaler = joblib.load('rent_minmax_scaler.joblib')
print("Scaler feature names:")
print(scaler.feature_names_in_)
