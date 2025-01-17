import numpy as np
import pandas as pd

df = pd.read_csv('psc_severity_train.csv')


missing_data_rows = df[df.isnull().any(axis=1)]
# Drop rows with missing data
filtered_df = df.dropna()

# Save the filtered data to a new CSV file
filtered_df.to_csv('filtered_data.csv', index=False)
missing_data_rows.to_csv('missing_data.csv', index=False)

print(filtered_df)
print(missing_data_rows)