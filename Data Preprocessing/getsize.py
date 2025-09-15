import pandas as pd
import sys
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np

# Example usage
filepath = 'dbNSFP4.9a_variant_combined.csv'  # Replace with your actual file name
chunksize = 10000  # Adjust as needed

# --- Start EDA Output ---
with open('dbnsfp_eda_results.txt', 'w') as f:
    # --- Initialize variables for accumulating results across chunks ---
    total_rows = 0
    total_cols = 0
    total_size = 0
    all_missing_values = {}
    all_outlier_counts = {}
    dropped_columns = set()
    processed_chunks = []  # Create an empty list to store processed chunks

    # Load the DataFrame in chunks
    chunks = pd.read_csv(filepath, na_values=['.', '.;.', '.;.;', './.', ',',' .','. ',' . ','na','NA','-','-1','.;1/1;'], chunksize=chunksize, low_memory=False)

    for chunk in chunks:
        # Data matrix size
        rows, cols = chunk.shape
        total_rows += rows
        total_cols = cols  # Assuming all chunks have the same number of columns initially
        total_size += sys.getsizeof(chunk)

        # --- Missing Value Analysis ---

        # i) Find the number of missing values for each feature
        missing_values = chunk.isnull().sum()
        for col, count in missing_values.items():
            all_missing_values[col] = all_missing_values.get(col, 0) + count

        # Get initial column names (only in the first chunk)
        if total_rows == rows:
            initial_columns = set(chunk.columns)

        # ii) Remove features with less than 80% values
        threshold = 0.8 * len(chunk)
        chunk = chunk.dropna(thresh=threshold, axis=1)

        # Identify dropped columns
        dropped_columns.update(initial_columns - set(chunk.columns))

        # Select only numeric features for imputation
        numeric_features = chunk.select_dtypes(include=np.number).columns

        # iii) Impute missing values for features with less than 20% missing values
        imputer = KNNImputer(n_neighbors=5)
        missing_below_20 = missing_values[missing_values < 0.2 * len(chunk)].index
        # Impute only on numeric features
        numeric_missing_below_20 = missing_below_20.intersection(numeric_features)
        chunk[numeric_missing_below_20] = imputer.fit_transform(chunk[numeric_missing_below_20])

        # --- Outlier Analysis ---

        # i) Define outliers (using IQR method)
        def find_outliers_iqr(data):
            if pd.api.types.is_numeric_dtype(data):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (data < lower_bound) | (data > upper_bound)
            else:
                return pd.Series(False, index=data.index)

        # ii) Causes for outliers ...

        # iii) Remove outliers
        outliers = chunk.apply(find_outliers_iqr, axis=0)
        chunk = chunk[~outliers.any(axis=1)]

        # v) Count outliers for each feature
        outlier_counts = outliers.sum()
        for col, count in outlier_counts.items():
            all_outlier_counts[col] = all_outlier_counts.get(col, 0) + count

        # Append the processed chunk to the list
        processed_chunks.append(chunk)

    # Concatenate the processed chunks into a single DataFrame
    dbnsfp_data = pd.concat(processed_chunks, ignore_index=True)

    # --- Write accumulated results to the output file ---

    f.write(f"dbNSFP data matrix size: {total_rows} rows x {total_cols} columns\n")
    f.write(f"Total size of dbNSFP DataFrame (approximate): {total_size} bytes\n\n")

    f.write("Missing values per feature:\n")
    for col, count in all_missing_values.items():
        f.write(f"{col}: {count}\n")

    f.write(f"\nDropped columns: {list(dropped_columns)}\n")

    removed_features_count = len(initial_columns) - len(dbnsfp_data.columns)
    f.write(f"\nNumber of features removed during missing value analysis: {removed_features_count}\n\n")

    f.write("Outlier counts per feature:\n")
    for col, count in all_outlier_counts.items():
        f.write(f"{col}: {count}\n")

    cleaned_rows, cleaned_cols = dbnsfp_data.shape
    f.write(f"\n\nCleaned dbNSFP data matrix size: {cleaned_rows} rows x {cleaned_cols} columns\n")