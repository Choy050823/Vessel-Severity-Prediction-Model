import numpy as np
import pandas as pd

df = pd.read_csv('psc_severity_train.csv')

# Gets the name of the column 0 (PscInspectionId)
column_name = df.columns[0]

# Count occurrences of each value in column 0 (PscInspectionId)
value_counts = df[column_name].value_counts()

# Identify values that do not appear exactly 3 times
valid_values = value_counts[value_counts == 3].index
invalid_values = value_counts[value_counts != 3].index

# Filter rows where olumn 0 (PscInspectionId) has invalid occurrences
filtered_df = df[df[column_name].isin(valid_values)]
irregular_df = df[df[column_name].isin(invalid_values)]

# Save the filtered rows to a new CSV
filtered_df.to_csv('1.filtered_rows.csv', index=False)
irregular_df.to_csv('1.irregular_data.csv', index=False)

# Display the filtered DataFrame
print(filtered_df.isnull)
print(irregular_df.isnull)