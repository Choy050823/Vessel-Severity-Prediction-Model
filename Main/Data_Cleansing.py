import os
import pandas as pd
import numpy as np

# Split def_text into columns
df = pd.read_csv('psc_severity_train.csv', dtype={'deficiency_code': str})
df['inspection_id'] = df['def_text'].str.extract(r'PscInspectionId: (\d+)')
df['deficiency_finding'] = df['def_text'].str.extract(r'Deficiency/Finding: (.*?)\n', expand=False)
df['description_overview'] = df['def_text'].str.extract(r'Description Overview: (.*?)\n', expand=False)
df['immediate_causes'] = df['def_text'].str.extract(r'Immediate Causes: (.*?)\n', expand=False)
df['root_cause_analysis'] = df['def_text'].str.extract(r'Root Cause Analysis: (.*?)\n', expand=False)
df['corrective_action'] = df['def_text'].str.extract(r'Corrective Action: (.*?)\n', expand=False)
df['preventive_action'] = df['def_text'].str.extract(r'Preventive Action: (.*?)\n', expand=False)
df['deficiency_code'] = df['def_text'].str.extract(r'Deficiency Code: (\d+)', expand=False)
df['detainable_deficiency'] = df['def_text'].str.extract(r'Detainable Deficiency: (\w+)', expand=False)

# Drop columns 'def_text' and 'inspection_id'
df = df.drop('def_text', axis=1)
df = df.drop('inspection_id', axis=1)

# Add a zero for 4-digit deficiency_code
df['deficiency_code'] = df['deficiency_code'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)
df['deficiency_code'] = df['deficiency_code'].astype(str)

# Locate missing data rows
missing_data_rows = df[df.isnull().any(axis=1)]

# Drop rows with missing data
filtered_df = df.dropna()

# Drop irrelevant columns
df_final = filtered_df.drop(filtered_df.columns[[0,2,3,6,7]], axis=1)

# Save the new DataFrame to a new CSV file
df_final.to_csv(os.path.join('Cleansed_Data', 'cleansed_data.csv'), index=False)

# Display the filtered DataFrame
print(df_final)