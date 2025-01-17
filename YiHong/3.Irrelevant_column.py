import numpy as np
import pandas as pd

df = pd.read_csv('psc_severity_train.csv')

# Select columns 1 (deficiency_code), 4 (annotation_severity), 5 (def_text), 8(PscAuthorityId), 9(PortId), 10 (VesselGroup) and 11 (age)
selected_columns = [1, 4, 5, 8, 9, 10, 11]
unselected_columns = [0, 2, 3, 6, 7]

# Create a new DataFrame with the selected columns
df_selected = df.iloc[:, selected_columns]
df_unselected = df.iloc[:, unselected_columns]

# Save the new DataFrame to a new CSV file
df_selected.to_csv('3.selected_data.csv', index=False)
df_unselected.to_csv('3.unselected_data.csv', index=False) 

# Display the filtered DataFrame
print(df_selected)
print(df_unselected)