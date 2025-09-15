import dask.dataframe as dd

# Define input and output file paths
input_filepath = 'variant_summary.csv'
output_filepath = 'filtered_clinvar.csv'

# Read the ClinVar CSV file using Dask, specifying the dtype for 'Chromosome'
clinvar_data = dd.read_csv(input_filepath, dtype={'Chromosome': 'object'})

# Select the columns to keep
columns_to_keep = ['ClinicalSignificance', 'ReviewStatus', 'Chromosome', 'Start',
                   'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Name']
filtered_clinvar_data = clinvar_data[columns_to_keep]

# Write the filtered data to a new CSV file
filtered_clinvar_data.to_csv(output_filepath, single_file=True, index=False)