"""Quickly inspect the contents and completeness of a CSV file."""

import pandas as pd


csv_path = "path/to/your_file.csv"  # Replace with the CSV file path.
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print("\nRows and columns:", df.shape)
print("\nAvailable (non-empty) values per column:")
print(df.count().sort_values(ascending=False))
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))

# Preview rows sorted by each column (where the values are sortable).
for column in df.columns:
	print(f"\nSorted by {column}:")
	print(df.sort_values(by=column, na_position="last").head())
