import joblib

mean_encoder = joblib.load('rent_mean_encoder.joblib')

features_to_check = ['cityname', 'state', 'source']

for feature in features_to_check:
    if hasattr(mean_encoder, 'categories_'):
        try:
            idx = mean_encoder.feature_names_in_.tolist().index(feature)
            print(f"Categories for {feature}:")
            print(mean_encoder.categories_[idx])
        except Exception as e:
            print(f"Error getting categories for {feature}: {e}")
    else:
        print(f"No categories_ attribute in mean_encoder for {feature}")
