import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Load the dataset and perform basic preprocessing.
    """
    # Load the dataset
    df = pd.read_csv(filepath)

    # Filter relevant columns
    relevant_columns = [
        'deficiency_finding', 
        'description_overview', 
        'immediate_causes', 
        'root_cause_analysis',
        'VesselGroup',
        'detainable_deficiency',
        'age',
        'adjusted_severity'
    ]
    df = df[relevant_columns]

    # Drop rows with missing values
    df = df.dropna(subset=relevant_columns)

    return df