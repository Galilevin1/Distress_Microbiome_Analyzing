import pandas as pd
import matplotlib.pyplot as plt
import MIPMLP


# Save overlap samples only between Data and Metadata and arrange it by the same order
def save_overlap_samples_and_arrange_by_order(processed_path, tag_path, folder, Data_name, tag_name, add_taxonomy_for_preprocess=False):
    """

    Parametesr:
    - processed_path (str): Directory of the "preprocess"/"MIPMLP" data CSV file
    - path_tag (str): Directory of the "metadata" CSV file
    - folder (str): Directory for saving output files
    - Data_name (str): Data file name for saving
    - tag_name (str): Tag file name for saving
    - add_taxonomy_for_preprocess (bool): True, will add a taxonomy row as the last row in the csv file. Default is False.
    For "preprocess" csv file should be set to true. For "MIPMLP" type csv file should be set to False.

    return:
    -  df_data_sorted (pd.DataFrame): A new DataFrame for "preprocess" or "MIPMLP" data, containing overlapping
     and same order values with df_tag.
     - df_tag(pd.DataFrame): A new DataFrame for "metadata", containing overlapping and same order values
      with df_tag.
    """
    def compare_and_save(file1, file2):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        # Extract the first column from each DataFrame
        column1_file1 = df1.iloc[:, 0]
        column1_file2 = df2.iloc[:, 0]
        # Find the overlapping values
        overlapping_values = column1_file1[column1_file1.isin(column1_file2)]
        # Filter rows based on overlapping values
        df1_overlapping = df1[df1.iloc[:, 0].isin(overlapping_values)]
        if add_taxonomy_for_preprocess:
           df1_overlapping = df1_overlapping._append(df1.iloc[-1], ignore_index=True)
        df2_overlapping = df2[df2.iloc[:, 0].isin(overlapping_values)]
        return df1_overlapping, df2_overlapping

    # Save only overlap samples between 2 CSV files
    df_data, df_tag = compare_and_save(processed_path, tag_path)

    # Arrange samples in both tag and preprocess files to be at the same order
    index_mapping = {value: index for index, value in enumerate(df_tag.iloc[:, 0])}
    df_data_sorted = df_data.iloc[df_data.iloc[:, 0].map(index_mapping).argsort()]

    # # Set the first column as the index for both DataFrames
    # df_data_sorted.set_index(df_data_sorted.columns[0], inplace=True)
    # df_tag.set_index(df_tag.columns[0], inplace=True)
    #
    # # Save the resulting DataFrames to CSV files
    # df_data_sorted.to_csv(processed_overlap_path, index=True)
    # df_tag.to_csv(tag_overlap_path, index=True)

    # Save the resulting DataFrames to CSV files
    df_data_sorted.to_csv(f"{folder}/{Data_name}_ordered.csv", index=False)
    df_tag.to_csv(f"{folder}/{tag_name}_ordered.csv", index=False)

    return df_data_sorted, df_tag

def Data_quality_check(df_preprocess, df_MIPMLP, df_metadata, folder, preprocess_saving_name, MIPMLP_saving_name, tag_saving_name, non_zero_treshold_cols=4, non_zero_treshold_rows=4, non_zero_threshold_MIPMLP_cols=4):

    """

    Parameters:
    - df_preprocess(pd.DataFrame): "Preprocess" DataFrame.
    - df_MIPMLP(pd.DataFrame): "MIPMLP" DataFrame.
    - df_metadata(pd.DataFrame): Metadata DataFrame.
    - folder: Directory for input files and saving the output files.
    - non_zero_treshold_cols (int, optional): Threshold for the count of non-zeros microbes frequencies in the "preprocess" data.
    - preprocess_saving_name (str): Preprocess file name for saving
    - MIPMLP_saving_name (str): MIPMLP file name for saving
    - tag_saving_name (str): Tag file name for saving
     Count lower than threshold will be later dropped from data.
    - non_zero_treshold_rows( int, optional): Threshold for the count of non-zeros samples frequencies.
     Count lower than threshold will be later dropped from data.
    - non_zero_threshold_MIPMLP_cols: Threshold for the count of non-zeros microbes frequencies in the "MIPMLP" data.
     Count lower than threshold will be later dropped from data.

    return:
    - df_MIPMLP (pd.DataFrame): "MIPMLP" data after dropping the non-quality microbes and sample.
    - df_preprocess (pd.DataFrame): "Preprocess" file after dropping the corresponding samples as dropped on the "MIPMLP" file.
    - df_metadata (pd.DataFrame): Metadata file after dropping the corresponding samples as dropped on the "MIPMLP" file.

    """

    def preprocess_quality_check(df_preprocess, non_zero_treshold_cols, non_zero_treshold_rows):

        # Ensure the DataFrame columns are numeric

        last_row = df_preprocess.iloc[-1:]
        df_preprocess = df_preprocess.iloc[:-1].apply(pd.to_numeric, errors='coerce')
        df_preprocess = pd.concat([df_preprocess, last_row], ignore_index=False)

        # Step 1: For each microbe, how many samples are not 0

        # Count the number of non-zero entries for each column
        non_zero_counts_per_col = (df_preprocess.iloc[:-1] != 0).sum(axis=0)
        microbe_names = df_preprocess.iloc[-1]
        microbe_names_list = microbe_names.values.tolist()
        # Extract feature with low count of samples: indices and corresponding feature names
        low_count_col_indices = [i for i, count in enumerate(non_zero_counts_per_col) if count <= non_zero_treshold_cols]
        low_count_features = [microbe_names_list[i] for i in low_count_col_indices]
        samples_counts = {}
        for count in non_zero_counts_per_col:
            samples_counts.setdefault(count, 0)
            samples_counts[count] += 1
        # Plot histogram
        plt.bar(samples_counts.keys(), samples_counts.values(), color='skyblue')
        plt.xlabel('Non-zero samples in a microbe')
        plt.ylabel('Number of Microbes')
        plt.title('Distribution of Microbes Across non-zero Samples')
        plt.grid(True)
        plt.show()

        # Step 2: For each sample, how many features are not 0

        # Count the number of non-zero features for each sample (row)
        non_zero_count_per_row = (df_preprocess.iloc[:-1] != 0).sum(axis=1)  # .iloc[:, :-1] ??
        # Extract feature with low count of samples: indices and corresponding feature names
        low_count_row_indices = [i for i, count in enumerate(non_zero_count_per_row) if count < non_zero_treshold_rows]
        features_counts = {}
        for count in non_zero_count_per_row:
            features_counts.setdefault(count, 0)
            features_counts[count] += 1
        # Plot histogram
        plt.bar(features_counts.keys(), features_counts.values(), color='skyblue')
        plt.xlabel('Non-zero microbes in a sample')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Samples Across non-zero Microbes')
        plt.grid(True)
        plt.show()
        # Print samples lower than the threshold
        print("Samples with non-zero counts lower than the threshold:")
        if len(low_count_row_indices)==0:
          print("All samples passed quality checks")
        for sample_index in low_count_row_indices:
            print(f"Sample {sample_index}")
        return low_count_row_indices

    # Step 3: check MIPMLP characteristics

    def MIPMLP_qualiy_check(df_MIPMLP, non_zero_threshold_MIPMLP_cols):
        # Define parameters
        filtered_col_indices = []
        filtered_col_names = []
        # Iterate through each column
        for idx, column in enumerate(df_MIPMLP.columns):
            # Get the most frequent value and its count
            most_frequent_value = df_MIPMLP[column].mode()[0]
            most_frequent_count = (df_MIPMLP[column] == most_frequent_value).sum()
            # Calculate the number of unique values excluding the most frequent one
            unique_values = df_MIPMLP[column].nunique() - 1
            # Check if the unique values count is below the threshold
            if unique_values < non_zero_threshold_MIPMLP_cols:
                filtered_col_indices.append(idx)
                filtered_col_names.append(column)

        # Print the results
        print("Columns with unique values below the threshold (excluding the most frequent value):")
        print("Amount:", len(filtered_col_indices))
        print("Indices:", filtered_col_indices)
        print("Names:", filtered_col_names)
        return filtered_col_indices, filtered_col_names

    # Step 4: drop the non-quality columns and rows

    def drop_cols_rows(df_MIPMLP, df_preprocess, df_metadata, preprocess_saving_path, MIPMLP_saving_path, tag_saving_path, columns_to_drop = [], rows_to_drop = [], condition = "", MIPMLP_mean= ""):
        # Drop problematic columns
        df_MIPMLP = df_MIPMLP.drop(columns=columns_to_drop)
        # Drop problematic rows
        df_MIPMLP = df_MIPMLP.drop(index=rows_to_drop)
        df_metadata = df_metadata.drop(index=rows_to_drop)
        df_preprocess = df_preprocess.drop(index=rows_to_drop)
        # Reset the index after dropping rows
        df_MIPMLP.reset_index(drop=True, inplace=True)
        df_metadata.reset_index(drop=True, inplace=True)
        df_preprocess.reset_index(drop=True, inplace=True)
        # Save files
        df_MIPMLP.to_csv(f"{folder}/MIPMLP_mean_filtered.csv", index=False)
        df_preprocess.to_csv(f"{folder}/preprocess_filtered.csv", index=False)
        df_metadata.to_csv(f"{folder}/metadata_filtered.csv", index=False)
        return df_MIPMLP, df_preprocess, df_metadata

    # Reset the index after dropping rows
    df_MIPMLP.reset_index(drop=True, inplace=True)
    df_metadata.reset_index(drop=True, inplace=True)
    df_preprocess.reset_index(drop=True, inplace=True)

    low_count_row_indices = preprocess_quality_check(df_preprocess, non_zero_treshold_cols, non_zero_treshold_rows)
    filtered_col_indices, filtered_col_names = MIPMLP_qualiy_check(df_MIPMLP, non_zero_threshold_MIPMLP_cols)
    df_MIPMLP, df_preprocess, df_metadata = drop_cols_rows(df_MIPMLP, df_preprocess, df_metadata, preprocess_saving_name, MIPMLP_saving_name, tag_saving_name, columns_to_drop=filtered_col_names, rows_to_drop=low_count_row_indices, condition="", MIPMLP_mean="")
    return df_MIPMLP, df_preprocess, df_metadata

def apply_preprocess(preprocess_path, tag_path, folder,  MIPMLP_saving_name, preprocess_saving_name, tag_saving_name, normalization='relative'):

    """
    Apply pre-process to 3 CSV file: "preprocess", "MIPMLP", and metadata. Saving only overlapping samples, arrange the
    files by the same order of samples, and finally perform quality check to the microbes and samples and save only the ones who passed.

    Parameters:
    - preprocess_path (str): Directory of the "preprocess" data CSV file
    - tag_path (str): Directory of the "metadata" CSV file
    - folder (str): Directory for saving output files
    - MIPMLP_saving_name (str): Name for saving "MIPMLP" relates DataFrames
    - preprocess_saving_name (str): Name for saving "preprocess" relates DataFrames
    - tag_saving_name (str): Name for saving "metadata" DataFrames
    - normalization (str): Normalization type for MIPMLP. Deafult is 'relative'.

    Returns:
  - None. Saves output files to the specified folder.
    """

    df_preprocess = pd.read_csv(preprocess_path)
    df_MIPMLP = MIPMLP.preprocess(df_preprocess, normalization=normalization)
    #df_MIPMLP = MIPMLP.preprocess(df_preprocess, taxnomy_group="sub PCA")
    df_MIPMLP.rename_axis("ID", inplace=True)
    df_MIPMLP.to_csv(f"{folder}/{MIPMLP_saving_name}.csv", index=True)
    df_preprocess_ordered, df_metadata_ordered = save_overlap_samples_and_arrange_by_order(preprocess_path, tag_path, folder, preprocess_saving_name, tag_saving_name, add_taxonomy_for_preprocess=True)
    df_MIPMLP_ordered, df_metadata_ordered = save_overlap_samples_and_arrange_by_order(f"{folder}/{MIPMLP_saving_name}.csv", tag_path, folder, MIPMLP_saving_name, tag_saving_name, add_taxonomy_for_preprocess=False)
    df_MIPMLP_quality, df_preprocess_quality, df_metadata_quality = Data_quality_check(df_preprocess_ordered, df_MIPMLP_ordered, df_metadata_ordered, folder, preprocess_saving_name, MIPMLP_saving_name, tag_saving_name, non_zero_treshold_cols=4, non_zero_treshold_rows=4, non_zero_threshold_MIPMLP_cols=4)

    return

####################################### Running #######################################

# Defining parameters
# Files' names
project = "Distress_Projects_Analysis"
folder = f"Datas/{project}"
# Input path
preprocess_path = f"{folder}/for_preprocess.csv"
tag_path = f"{folder}/metadata.csv"
# Output path
MIPMLP_saving_name = f"MIPMLP_scaled_adj"
preprocess_saving_name = f"for_preprocess"
tag_saving_name = f"metadata_adj"

# Apply Data processing
apply_preprocess(preprocess_path, tag_path, folder, MIPMLP_saving_name, preprocess_saving_name, tag_saving_name)

########################################################################################################################
