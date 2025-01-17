import pandas as pd

# Load the CSV files
df1 = pd.read_csv('./Main/Cleansed_Data/predicted_severity.csv')
df2 = pd.read_csv('./Main/Cleansed_Data/augmented_data2.csv')

# Append the two dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('./Main/Cleansed_Data/final_data.csv', index=False)
print(combined_df.head())