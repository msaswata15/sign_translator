import pandas as pd
import os

# Paths
labels_csv_path = "data/raw/labels.csv"  # replace with your actual path
processed_data_dir = "data/processed"  # directory containing processed files
output_csv_path = "data/raw/labels.csv"  # can overwrite labels_csv_path if you want

# Load labels
df = pd.read_csv(labels_csv_path, sep='\t')  # use sep='\t' if it's tab-separated

# Get list of processed file names (without extension)
processed_files = {os.path.splitext(f)[0] for f in os.listdir(processed_data_dir)}

# Filter only rows where SENTENCE_NAME exists in processed files
df_cleaned = df[df['SENTENCE_NAME'].isin(processed_files)]

# Save the cleaned dataframe
df_cleaned.to_csv(output_csv_path, sep='\t', index=False)

print(f"Cleaned labels saved to: {output_csv_path}")
print(f"Original count: {len(df)}, After cleaning: {len(df_cleaned)}, Removed: {len(df) - len(df_cleaned)}")
