import pandas as pd
from scipy.stats import mode

# Load the dataset
df = pd.read_csv('psc_severity_train.csv')

# Handle missing annotations (drop rows with missing severity)
df = df.dropna(subset=['annotation_severity'])

# Group by deficiency and derive consensus severity
grouped = df.groupby(['PscInspectionId', 'deficiency_code'])

# error here
def get_consensus_severity(group):
    return mode(group['annotation_severity']).mode[0]

consensus = grouped.apply(get_consensus_severity).reset_index()
consensus.columns = ['PscInspectionId', 'deficiency_code', 'consensus_severity']

# Merge consensus severity back into the original dataset
df = pd.merge(df, consensus, on=['PscInspectionId', 'deficiency_code'], how='left')

# Save the dataset with consensus severity
df.to_csv('train_dataset_with_consensus.csv', index=False)
print("Hello")