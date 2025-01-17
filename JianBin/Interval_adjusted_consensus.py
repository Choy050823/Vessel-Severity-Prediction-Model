# this is the interval code that works
import pandas as pd
import numpy as np

def calculate_stats_and_adjusted_severity(df):
    # Create severity mapping
    severity_map = {
        'Low': 1,
        'Medium': 2,
        'High': 3
    }

    # Convert severity to numerical values
    df['severity_numeric'] = df['annotation_severity'].map(severity_map)

    # Initialize dictionaries
    consensus_dict = {}
    std_dict = {}
    adjusted_dict = {}

    # Group by VesselId
    for vessel_id in df['VesselId'].unique():
        # Get all entries for this vessel
        vessel_group = df[df['VesselId'] == vessel_id]

        # Calculate consensus
        severity_sum = vessel_group['severity_numeric'].sum()
        num_experts = len(vessel_group)
        vessel_consensus = round(severity_sum / num_experts, 2)

        # Calculate standard deviation
        vessel_std = round(vessel_group['severity_numeric'].std(), 2)
        if pd.isna(vessel_std):  # Handle case where there's only one rating
            vessel_std = 0.0

        # Calculate adjusted severity based on new std threshold (0.65)
        if vessel_consensus < 1.66:
            # Round up
            adjusted_severity = 1 
        elif vessel_consensus <= 2.33:
            adjusted_severity = 2
        elif vessel_consensus < 3.00:
            adjusted_severity = 3
        else:
            adjusted_severity = 3
            # Round to nearest integer
            adjusted_severity = round(vessel_consensus)

        # Store values
        consensus_dict[vessel_id] = vessel_consensus
        std_dict[vessel_id] = vessel_std
        adjusted_dict[vessel_id] = adjusted_severity

        # Print debug info
        print(f"\nVessel ID: {vessel_id}")
        print(f"Number of experts: {num_experts}")
        print(f"Consensus: {vessel_consensus}")
        print(f"Standard Deviation: {vessel_std}")
        print(f"Adjusted Severity: {adjusted_severity}")

    # Add columns to original dataframe
    df['severity_consensus'] = df['VesselId'].map(consensus_dict)
    df['severity_std'] = df['VesselId'].map(std_dict)
    df['adjusted_severity'] = df['VesselId'].map(adjusted_dict)

    return df

# Read the training data
input_file = '/content/psc_severity_train.csv'
df = pd.read_csv(input_file)

# Calculate all statistics
df_with_stats = calculate_stats_and_adjusted_severity(df)

# Save the result
output_file = '/content/psc_severity_train_with_interval_adjusted.csv'
df_with_stats.to_csv(output_file, index=False)

# Print summary statistics
print("\nSummary:")
print(f"Total rows processed: {len(df)}")
print(f"Number of unique vessels: {df['VesselId'].nunique()}")
print(f"Results saved to: {output_file}")

# Display sample results grouped by VesselId
print("\nSample results with new threshold:")
sample_vessels = df_with_stats['VesselId'].unique()[:5]  # First 5 vessels
for vessel in sample_vessels:
    vessel_data = df_with_stats[df_with_stats['VesselId'] == vessel]
    print(f"\nVessel ID: {vessel}")
    print(vessel_data[['username', 'annotation_severity', 'severity_numeric',
                      'severity_consensus', 'severity_std', 'adjusted_severity']].to_string())

# Print statistics about adjustments with new threshold
total_vessels = df_with_stats['VesselId'].nunique()
high_std_vessels = df_with_stats[df_with_stats['severity_std'] > 0.65]['VesselId'].nunique()
print(f"\nVessels with high standard deviation (>0.65): {high_std_vessels} out of {total_vessels}")
print(f"Vessels with low severity (1): {df_with_stats[df_with_stats['adjusted_severity'] == 1]['VesselId'].nunique()} out of {total_vessels}")
print(f"Vessels with medium severity (2): {df_with_stats[df_with_stats['adjusted_severity'] == 2]['VesselId'].nunique()} out of {total_vessels}")
print(f"Vessels with high severity (3): {df_with_stats[df_with_stats['adjusted_severity'] == 3]['VesselId'].nunique()} out of {total_vessels}")
