import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt

# File path
file_path = 'filtered_linked_data.csv'
chunk_size = 10000  # Define the chunk size

#Missing Values list
na_list = [
    '#chr', 'pos(1-based)', 'ref', 'alt',
    'SIFT_converted_rankscore', 'SIFT4G_converted_rankscore', 'Polyphen2_HDIV_rankscore',
    'Polyphen2_HVAR_rankscore', 'LRT_converted_rankscore', 'MutationTaster_converted_rankscore',
    'MutationAssessor_rankscore', 'FATHMM_converted_rankscore', 'PROVEAN_converted_rankscore',
    'VEST4_rankscore', 'MetaSVM_rankscore', 'MetaLR_rankscore', 'MetaRNN_rankscore', 'M-CAP_rankscore',
    'REVEL_rankscore', 'MutPred_rankscore', 'MVP_rankscore', 'gMVP_rankscore', 'MPC_rankscore',
    'PrimateAI_rankscore', 'DEOGEN2_rankscore', 'BayesDel_addAF_rankscore', 'BayesDel_noAF_rankscore',
    'ClinPred_rankscore', 'LIST-S2_rankscore', 'VARITY_R_rankscore', 'VARITY_ER_rankscore',
    'VARITY_R_LOO_rankscore', 'VARITY_ER_LOO_rankscore', 'ESM1b_rankscore', 'EVE_rankscore',
    'AlphaMissense_rankscore', 'PHACTboost_rankscore', 'MutFormer_rankscore', 'MutScore_rankscore',
    'CADD_raw_rankscore', 'CADD_raw_rankscore_hg19', 'DANN_rankscore', 'fathmm-MKL_coding_rankscore',
    'fathmm-XF_coding_rankscore', 'Eigen-raw_coding_rankscore', 'Eigen-PC-raw_coding_rankscore',
    'GenoCanyon_rankscore', 'integrated_fitCons_rankscore', 'GM12878_fitCons_rankscore',
    'H1-hESC_fitCons_rankscore', 'HUVEC_fitCons_rankscore', 'LINSIGHT_rankscore', 'GERP++_RS_rankscore',
    'GERP_91_mammals_rankscore', 'phyloP100way_vertebrate_rankscore', 'phyloP470way_mammalian_rankscore',
    'phyloP17way_primate_rankscore', 'phastCons100way_vertebrate_rankscore', 'phastCons470way_mammalian_rankscore',
    'phastCons17way_primate_rankscore', 'SiPhy_29way_logOdds_rankscore', 'bStatistic_converted_rankscore', '.',''
]

# Initialize variables to aggregate results across chunks
missing_values_summary = None
data_cleaned_chunks = []
outlier_counts_summary = {}

# Define outlier detection function
def detect_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers

# Process the data chunk by chunk
for chunk in pd.read_csv(file_path, na_values=na_list, chunksize=chunk_size):
    print(f"Processing chunk with shape: {chunk.shape}")

    # Set dtypes of index features to object
    index_features = ['variant_id', 'ClinicalSignificance']
    for feature in index_features:
        if feature in chunk.columns:
            chunk[feature] = chunk[feature].astype('object')

    if 'variant_id' in chunk.columns:
        chunk = chunk[chunk['variant_id'].notnull()]
        print(f"Chunk shape after removing rows with missing 'variant_id': {chunk.shape}")

    # Separate non-index features for EDA
    non_index_features = [col for col in chunk.columns if col not in index_features]
    chunk_non_index = chunk[non_index_features]

    # Count missing values per column in the chunk
    missing_values_chunk = chunk_non_index.isnull().sum()

    # Aggregate missing values across chunks
    if missing_values_summary is None:
        missing_values_summary = missing_values_chunk
    else:
        missing_values_summary += missing_values_chunk

    # Remove features with >= 80% missing values
    threshold = 0.8
    features_to_keep = missing_values_chunk[missing_values_chunk < threshold * chunk.shape[0]].index
    chunk_non_index = chunk_non_index[features_to_keep]

    # Impute missing values for features with <20% missing values
    imputer = KNNImputer(n_neighbors=5)
    chunk_imputed = pd.DataFrame(imputer.fit_transform(chunk_non_index), columns=chunk_non_index.columns)

    # Reattach index features
    chunk_imputed[index_features] = chunk[index_features].reset_index(drop=True)

    numeric_features = chunk_imputed.select_dtypes(include='number').columns
    scaler = RobustScaler()
    chunk_imputed[numeric_features] = scaler.fit_transform(chunk_imputed[numeric_features])

    # Detect and remove outliers for numeric columns
    for feature in chunk_imputed.select_dtypes(include='number').columns:
        Q1 = chunk_imputed[feature].quantile(0.25)
        Q3 = chunk_imputed[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR

        # Count outliers for reporting
        outliers = chunk_imputed[(chunk_imputed[feature] < lower_bound) | (chunk_imputed[feature] > upper_bound)]
        outlier_counts_summary[feature] = outlier_counts_summary.get(feature, 0) + len(outliers)

        # Remove outliers
        chunk_imputed = chunk_imputed[(chunk_imputed[feature] >= lower_bound) & (chunk_imputed[feature] <= upper_bound)]

    # Collect the cleaned chunk
    data_cleaned_chunks.append(chunk_imputed)

# Combine all cleaned chunks
cleaned_data = pd.concat(data_cleaned_chunks, ignore_index=True)

# Report results
removed_features = missing_values_summary[missing_values_summary >= threshold * chunk_size].index
outlier_count_total = sum(outlier_counts_summary.values())

report = []
report.append(f"Initial missing values per feature:\n{missing_values_summary}")
report.append(f"Number of features removed due to missing values: {len(removed_features)}")
report.append(f"Outlier counts per feature:\n{outlier_counts_summary}")
report.append(f"Number of samples removed during outlier analysis: {outlier_count_total}")
report.append(f"Final size of the cleaned dataset: {cleaned_data.shape}")

# Write report to a text file
with open('dbnsfp_eda_results.txt', 'w') as report_file:
    report_file.write("\n\n".join(report))

print("Data cleaning report saved to 'dbnsfp_eda_results.txt'")

# Visualize box plots
features_with_outliers = [feature for feature, count in outlier_counts_summary.items() if count > 0][:2]
features_without_outliers = [feature for feature, count in outlier_counts_summary.items() if count == 0][:2]

# Plot box plots for features with outliers
for feature in features_with_outliers:
    sns.boxplot(x=cleaned_data[feature])
    plt.title(f"Box Plot for Feature with Outliers: {feature}")
    plt.show()

# Plot box plots for features without outliers
for feature in features_without_outliers:
    sns.boxplot(x=cleaned_data[feature])
    plt.title(f"Box Plot for Feature without Outliers: {feature}")
    plt.show()

if 'ClinicalSignificance' in cleaned_data.columns:
    # Count the number of samples for each class
    class_counts = cleaned_data['ClinicalSignificance'].value_counts()

    # Print the class counts
    print("Number of samples for each class:")
    print(class_counts)
else:
    print("Target column 'ClinicalSignificance' is not found in the dataset!")

# Save the cleaned data
cleaned_data.to_csv('cleaned_linked_data.csv', index=False)
print("Cleaned data saved to 'cleaned_linked_data.csv'")

