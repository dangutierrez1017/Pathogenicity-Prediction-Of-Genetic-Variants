import dask.dataframe as dd

def link_datasets(dbnsfp_filepath, clinvar_filepath, output_filepath):
    """
    Links the filtered dbNSFP and ClinVar datasets based on common identifiers using Dask.

    Args:
    dbnsfp_filepath: Path to the filtered dbNSFP CSV file.
    clinvar_filepath: Path to the filtered ClinVar CSV file.
    output_filepath: Path to the output linked CSV file.
    """
    try:
        # Read the dbNSFP dataset using Dask
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
            'phyloP17way_primate_rankscore', 'phastCons100way_vertebrate_rankscore',
            'phastCons470way_mammalian_rankscore',
            'phastCons17way_primate_rankscore', 'SiPhy_29way_logOdds_rankscore', 'bStatistic_converted_rankscore', '.',
            ''
        ]
        dtype_dict = {'#chr': 'object', 'pos(1-based)': 'object', 'ref': 'object', 'alt': 'object'}
        dbnsfp_data = dd.read_csv(dbnsfp_filepath, na_values=na_list, dtype=dtype_dict)

        # Read the ClinVar dataset using Dask
        clinvar_data = dd.read_csv(clinvar_filepath,
            dtype={'Chromosome': 'object', 'PositionVCF': 'object', 'Start': 'float64'},
            na_values=['-1', 'na', '-', '.'])

        # Create a unique identifier in both datasets for merging
        dbnsfp_data['variant_id'] = (
            dbnsfp_data['#chr'].str.strip().astype(str) + '_' +
            dbnsfp_data['pos(1-based)'].str.strip().astype(str) + '_' +
            dbnsfp_data['ref'].str.strip() + '_' + dbnsfp_data['alt'].str.strip()
        )

        clinvar_data['variant_id'] = (
            clinvar_data['Chromosome'].str.strip().astype(str) + '_' +
            clinvar_data['PositionVCF'].astype(str).str.strip() + '_' +  # Convert to string first
            clinvar_data['ReferenceAlleleVCF'].str.strip() + '_' +
            clinvar_data['AlternateAlleleVCF'].str.strip()
        )

        print("dbNSFP DataFrame with variant_id:")
        print(dbnsfp_data[['#chr', 'pos(1-based)', 'ref', 'alt', 'variant_id']].head())
        print("\nClinVar DataFrame with variant_id:")
        print(clinvar_data[['Chromosome', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'variant_id']].head())


        # Select only the 'variant_id' and 'ClinicalSignificance' columns from the ClinVar dataset
        clinvar_data = clinvar_data[['variant_id', 'ClinicalSignificance']]

        # Merge the datasets using Dask
        linked_data = dd.merge(dbnsfp_data, clinvar_data, on='variant_id', how='inner')

        # Drop the specified columns after merging
        columns_to_drop = ['#chr', 'pos(1-based)', 'ref', 'alt']
        linked_data = linked_data.drop(columns_to_drop, axis=1)

        linked_data = linked_data[linked_data['ClinicalSignificance'].isin(['Pathogenic', 'Benign'])]
        linked_data = linked_data.dropna(subset=['ClinicalSignificance'])

        # Save the linked data to a new CSV file
        linked_data.to_csv(output_filepath, single_file=True, index=False)

    except FileNotFoundError:
        print(f"Error: Input file not found: {dbnsfp_filepath} or {clinvar_filepath}")
    except KeyError as e:
        print(f"Error: Column(s) not found for merging: {e}")
    except MemoryError:
        print("MemoryError: The operation ran out of memory. Consider increasing system memory or using smaller chunks.")

# Example usage
dbnsfp_filepath = 'filtered_dbnsfp.csv'
clinvar_filepath = 'filtered_clinvar.csv'
output_filepath = 'linked_data.csv'

link_datasets(dbnsfp_filepath, clinvar_filepath, output_filepath)




