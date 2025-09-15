import pandas as pd
import os

# Define input and output file paths
input_filepath = 'dbNSFP4.9a_variant_combined.csv'
output_filepath = 'filtered_dbnsfp.csv'

columns_to_keep = [
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
    'phastCons17way_primate_rankscore', 'SiPhy_29way_logOdds_rankscore', 'bStatistic_converted_rankscore'
]

# Read the CSV file in chunks
chunksize = 10000  # Adjust this value based on your available memory
with pd.read_csv(input_filepath,na_values=[""], usecols=columns_to_keep, chunksize=chunksize) as reader:
    for chunk in reader:
        # Process each chunk (if needed)

        # Append the chunk to the output file (writing in append mode)
        chunk.to_csv(output_filepath, mode='a', header=not os.path.exists(output_filepath), index=False)