import pandas as pd
import os
import pytest
from student_subject_intersection import pd as pd_module

def test_intersection_and_subject_update(tmp_path):
    # Create sample data for mat
    mat_data = {
        "school": ["GP", "GP", "MS"],
        "sex": ["F", "M", "F"],
        "age": [18, 17, 16],
        "address": ["U", "U", "R"],
        "famsize": ["GT3", "LE3", "LE3"],
        "Pstatus": ["A", "T", "T"],
        "Medu": [4, 3, 2],
        "Fedu": [4, 3, 2],
        "Mjob": ["at_home", "services", "other"],
        "Fjob": ["teacher", "other", "services"],
        "reason": ["course", "reputation", "home"],
        "nursery": ["yes", "no", "yes"],
        "internet": ["no", "yes", "yes"],
        "G1": [10, 12, 14],
        "G2": [11, 13, 15],
        "G3": [12, 14, 16]
    }
    mat_df = pd.DataFrame(mat_data)
    mat_file = tmp_path / "student-mat.csv"
    mat_df.to_csv(mat_file, sep=";", index=False)

    # Create sample data for por
    por_data = {
        "school": ["GP", "GP", "MS"],
        "sex": ["F", "M", "F"],
        "age": [18, 17, 15],
        "address": ["U", "U", "R"],
        "famsize": ["GT3", "LE3", "LE3"],
        "Pstatus": ["A", "T", "T"],
        "Medu": [4, 3, 2],
        "Fedu": [4, 3, 2],
        "Mjob": ["at_home", "services", "other"],
        "Fjob": ["teacher", "other", "services"],
        "reason": ["course", "reputation", "home"],
        "nursery": ["yes", "no", "yes"],
        "internet": ["no", "yes", "yes"],
        "G1": [9, 11, 13],
        "G2": [10, 12, 14],
        "G3": [11, 13, 15]
    }
    por_df = pd.DataFrame(por_data)
    por_file = tmp_path / "student-por.csv"
    por_df.to_csv(por_file, sep=";", index=False)

    # Load the datasets as in the script
    mat = pd.read_csv(mat_file, sep=";")
    por = pd.read_csv(por_file, sep=";")

    mat['subject'] = 'M'
    por['subject'] = 'P'

    id_cols = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]

    intersection = pd.merge(por[id_cols], mat[id_cols], how='inner')

    por.loc[por[id_cols].apply(tuple, axis=1).isin(intersection.apply(tuple, axis=1)), 'subject'] = 'B'
    mat.loc[mat[id_cols].apply(tuple, axis=1).isin(intersection.apply(tuple, axis=1)), 'subject'] = 'B'

    combined_df = pd.concat([por, mat], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=id_cols)

    # Check that students in intersection have subject 'B'
    for _, row in intersection.iterrows():
        mask = (combined_df[id_cols] == row.values).all(axis=1)
        subjects = combined_df.loc[mask, 'subject'].unique()
        assert 'B' in subjects

    # Check that combined_df has no duplicate students based on id_cols
    assert combined_df.duplicated(subset=id_cols).sum() == 0

    # Check that combined_df contains all students from both datasets
    assert len(combined_df) <= len(por) + len(mat)

if __name__ == "__main__":
    pytest.main()
