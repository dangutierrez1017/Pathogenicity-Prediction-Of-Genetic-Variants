import dask.dataframe as dd
import sys
from sklearn.impute import KNNImputer
import numpy as np

filepath = 'filtered_clinvar.csv'
output_filepath = 'clinvar_eda_results.txt'


clinvar_data = dd.read_csv(filepath,
                           dtype={'Chromosome': 'object','PositionVCF': 'float64','Start': 'float64'},
                           na_values= ['-1','na','-','.'])

# --- Start EDA Output ---
with open(output_filepath, 'w') as f:
    # Data matrix size
    shape = clinvar_data.shape
    rows, cols = shape[0].compute(), shape[1]
    f.write(f"ClinVar data matrix size: {rows} rows x {cols} columns\n")

    # --- Missing Value Analysis ---
    missing_values = clinvar_data.isnull().sum().compute()
    f.write("Missing values per feature (sorted):\n")
    for col, count in sorted(missing_values.items(), key=lambda item: item[1], reverse=True):
        f.write(f"{col}: {count}\n")

    initial_columns = set(clinvar_data.columns)
    threshold = 0.8 * len(clinvar_data)
    clinvar_data = clinvar_data.dropna(thresh=threshold)

    remaining_columns = set(clinvar_data.columns)
    dropped_columns = initial_columns.difference(remaining_columns)
    f.write(f"\nDropped columns: {list(dropped_columns)}\n")

    numeric_features = clinvar_data.select_dtypes(include=np.number).columns
    imputer = KNNImputer(n_neighbors=5)
    missing_below_20 = missing_values[missing_values < 0.2 * len(clinvar_data)].index
    # Check that the subset contains numeric columns
    numeric_missing_below_20 = missing_below_20.intersection(numeric_features)
    print("Columns to impute:", numeric_missing_below_20)

    # Check if we have columns to impute
    if len(numeric_missing_below_20) > 0:
        # Compute the subset before imputation
        df_subset = clinvar_data[numeric_missing_below_20].compute()

        # Impute missing values
        df_subset_imputed = imputer.fit_transform(df_subset)

        # Assign the imputed values back to the Dask DataFrame
        clinvar_data[numeric_missing_below_20] = df_subset_imputed
    else:
        print("No numeric columns with missing values below 20% to impute.")

    removed_features_count = len(missing_values) - len(clinvar_data.columns)
    f.write(f"\nNumber of features removed during missing value analysis: {removed_features_count}\n\n")

    shape = clinvar_data.shape
    cleaned_rows, cleaned_cols = shape[0].compute(), shape[1]
    f.write(f"\n\nCleaned ClinVar data matrix size: {cleaned_rows} rows x {cleaned_cols} columns\n")