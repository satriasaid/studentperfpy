import pandas as pd

df = pd.read_excel('Apartment For Rent.xlsx')

for col in ['cityname', 'state', 'source']:
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        print(f"Unique values for {col}:")
        print(unique_vals)
    else:
        print(f"Column {col} not found in the Excel file.")
