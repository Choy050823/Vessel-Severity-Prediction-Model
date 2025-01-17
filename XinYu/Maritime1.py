import pandas as pd
import numpy as np

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
# print(df[['inspection_id', 'deficiency_finding', 'description_overview', 'immediate_causes', 
#          'root_cause_analysis', 'corrective_action', 'preventive_action', 'deficiency_code', 'detainable_deficiency']].head())
df = df.drop('def_text', axis=1)

df['deficiency_code'] = df['deficiency_code'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)
df['deficiency_code'] = df['deficiency_code'].astype(str)
print(df[['deficiency_code']])
df.to_csv('Edited2.csv')

df.to_csv('Edited2.csv', index=False)


print(df)