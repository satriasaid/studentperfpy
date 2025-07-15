import pandas as pd

# Load the datasets
mat = pd.read_csv("student-mat.csv", encoding="UTF-8", sep=";")
por = pd.read_csv("student-por.csv", encoding="UTF-8", sep=";")

# Add subject columns
mat['subject'] = 'M'
por['subject'] = 'P'

# Columns to identify students (user specified subset)
id_cols = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]

# Find intersection of students based on id_cols
intersection = pd.merge(por[id_cols], mat[id_cols], how='inner')

# Mark subject as 'B' for students in intersection in both dataframes
por.loc[por[id_cols].apply(tuple, axis=1).isin(intersection.apply(tuple, axis=1)), 'subject'] = 'B'
mat.loc[mat[id_cols].apply(tuple, axis=1).isin(intersection.apply(tuple, axis=1)), 'subject'] = 'B'

# Combine the dataframes
combined_df = pd.concat([por, mat], ignore_index=True)

# Optional: drop duplicates if any
combined_df = combined_df.drop_duplicates(subset=id_cols)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("student_combined.csv", index=False, sep=';')

print("Combined dataframe with updated 'subject' column saved to 'student_combined.csv'.")
