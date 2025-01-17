import os
import numpy as np
import pandas as pd

df = pd.read_csv('psc_severity_train.csv')

# Locate missing data rows
missing_data_rows = df[df.isnull().any(axis=1)]

# Drop rows with missing data
filtered_df = df.dropna()

# Save the filtered data to a new CSV file
filtered_df.to_csv(os.path.join('Data', '2.filtered_data.csv'), index=False)
missing_data_rows.to_csv(os.path.join('Data', '2.missing_data.csv'), index=False)

# Display the filtered DataFrame
print('Data with no missing values')
print(filtered_df)
print('Data with missing values')
print(missing_data_rows)