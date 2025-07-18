import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved Random Forest model, scaler, mean encoder, and ordinal encoder
try:
    model = joblib.load('student_rf.joblib')
    scaler = joblib.load('minmax_scaler.joblib')
    mean_encoder = joblib.load('mean_encoder.joblib')
    ordinal_encoder = joblib.load('ordinal_encoder.joblib')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()


st.title("Student Pass Prediction App")

# Define input fields for student features
school = st.selectbox('School', ['GP', 'MS'])
sex = st.selectbox('Sex', ['F', 'M'])
age = st.number_input('Age', min_value=15, max_value=22, value=17)
address = st.selectbox('Address', ['U', 'R'])
famsize = st.selectbox('Family Size', ['LE3', 'GT3'])
Pstatus = st.selectbox('Parent Cohabitation Status', ['T', 'A'])
Medu = st.slider('Mother\'s Education', 0, 4, 2)
Fedu = st.slider('Father\'s Education', 0, 4, 2)
Mjob = st.selectbox('Mother\'s Job', ['teacher', 'health', 'services', 'at_home', 'other'])
Fjob = st.selectbox('Father\'s Job', ['teacher', 'health', 'services', 'at_home', 'other'])
reason = st.selectbox('Reason to Choose School', ['home', 'reputation', 'course', 'other'])
guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
traveltime = st.slider('Travel Time to School', 1, 4, 1)
studytime = st.slider('Weekly Study Time', 1, 4, 2)
failures = st.slider('Number of Past Failures', 0, 3, 0)
schoolsup = st.selectbox('Extra Educational Support', ['yes', 'no'])
famsup = st.selectbox('Family Educational Support', ['yes', 'no'])
paid = st.selectbox('Extra Paid Classes', ['yes', 'no'])
activities = st.selectbox('Extra-curricular Activities', ['yes', 'no'])
nursery = st.selectbox('Attended Nursery School', ['yes', 'no'])
higher = st.selectbox('Wants Higher Education', ['yes', 'no'])
internet = st.selectbox('Internet Access at Home', ['yes', 'no'])
romantic = st.selectbox('In a Romantic Relationship', ['yes', 'no'])
famrel = st.slider('Quality of Family Relationships', 1, 5, 3)
freetime = st.slider('Free Time After School', 1, 5, 3)
goout = st.slider('Going Out with Friends', 1, 5, 3)
Dalc = st.slider('Workday Alcohol Consumption', 1, 5, 1)
Walc = st.slider('Weekend Alcohol Consumption', 1, 5, 1)
health = st.slider('Current Health Status', 1, 5, 3)
absences = st.number_input('Number of School Absences', min_value=0, max_value=100, value=4)
G2 = st.slider('Second Period Grade (G2)', 0, 20, 10)
subject = st.selectbox('Subject', ['M', 'P'])

# Create initial input dictionary
input_dict = {
    'school': school,
    'sex': sex,
    'age': age,
    'address': address,
    'famsize': famsize,
    'Pstatus': Pstatus,
    'Medu': Medu,
    'Fedu': Fedu,
    'Mjob': Mjob,
    'Fjob': Fjob,
    'reason': reason,
    'guardian': guardian,
    'traveltime': traveltime,
    'studytime': studytime,
    'failures': failures,
    'schoolsup': schoolsup,
    'famsup': famsup,
    'paid': paid,
    'activities': activities,
    'nursery': nursery,
    'higher': higher,
    'internet': internet,
    'romantic': romantic,
    'famrel': famrel,
    'freetime': freetime,
    'goout': goout,
    'Dalc': Dalc,
    'Walc': Walc,
    'health': health,
    'absences': absences,
    'G2': G2,
    'subject': subject
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Apply ordinal encoding
try:
    if hasattr(ordinal_encoder, 'feature_names_in_'):
        ordinal_expected_columns = ordinal_encoder.feature_names_in_
        ordinal_columns_to_transform = [col for col in ordinal_expected_columns if col in input_df.columns]
        
        if ordinal_columns_to_transform:
            ordinal_subset = input_df[ordinal_columns_to_transform].copy()
            transformed_ordinal = ordinal_encoder.transform(ordinal_subset)
            input_df[ordinal_columns_to_transform] = transformed_ordinal
except Exception as e:
    st.error(f"Error with ordinal encoding: {str(e)}")

# Apply mean encoding to all columns together
try:
    if hasattr(mean_encoder, 'feature_names_in_'):
        mean_expected_columns = mean_encoder.feature_names_in_
        mean_columns_to_transform = [col for col in mean_expected_columns if col in input_df.columns]
        
        if mean_columns_to_transform:
            mean_subset = input_df[mean_columns_to_transform].copy()
            transformed_mean = mean_encoder.transform(mean_subset)
            input_df[mean_columns_to_transform] = transformed_mean
    else:
        # Fallback approach
        mean_encoded_columns = ['address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian','subject']
        available_mean_columns = [col for col in mean_encoded_columns if col in input_df.columns]
        
        if available_mean_columns:
            input_df[available_mean_columns] = mean_encoder.transform(input_df[available_mean_columns])
            
except Exception as e:
    st.error(f"Error with mean encoding: {str(e)}")
    st.write("Input DataFrame columns:", input_df.columns.tolist())

# Apply MinMax scaling to all columns except G1 and G3
try:
    if hasattr(scaler, 'feature_names_in_'):
        # Get the columns that the scaler was trained on
        scaler_expected_columns = scaler.feature_names_in_
        scaler_columns_to_transform = [col for col in scaler_expected_columns if col in input_df.columns]
        
        if scaler_columns_to_transform:
            # Transform all columns that the scaler expects
            input_df[scaler_columns_to_transform] = scaler.transform(input_df[scaler_columns_to_transform])
    else:
        # Fallback: scale all columns except G1 and G3
        columns_to_scale = [col for col in input_df.columns]
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
        
except Exception as e:
    st.error(f"Error with MinMax scaling: {str(e)}")
    st.write("Columns available for scaling:", input_df.columns.tolist())
    if hasattr(scaler, 'feature_names_in_'):
        st.write("Scaler expects these columns:", scaler.feature_names_in_)

# Prediction threshold
passing_rate = 2.5

if st.button('Predict'):
    try:
        prediction = model.predict(input_df)[0]
        if prediction >= passing_rate:
            st.success(f"The student is predicted to PASS")
        else:
            st.error(f"The student is predicted to FAIL")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Input DataFrame:")
        st.write(input_df)
        st.write("DataFrame shape:", input_df.shape)
        st.write("DataFrame columns:", input_df.columns.tolist())