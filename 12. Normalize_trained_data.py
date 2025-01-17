import os
import pandas as pd

# Load the CSV file
df = pd.read_csv(r'Data/8.adjusted_severity_data.csv')

# Specify the column name you want to process
column_name = 'adjusted_severity'

# Define a function to categorize the values
def categorize_value(value):
    if 0 <= value <= 1.666:
        return 'Low'
    elif 1.666 < value <= 2.333:
        return 'Medium'
    elif 2.333 < value <= 4:
        return 'High'
    else:
        return value  # If out of range, keep the original value (optional)

# Apply the categorization function to the specific column and replace the values
df[column_name] = df[column_name].apply(categorize_value)

# Save the result to a new CSV file
df.to_csv(os.path.join('Data', '12.normalized_data.csv'), index=False)

# Print the updated dataframe
print(df)