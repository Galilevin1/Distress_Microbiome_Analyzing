## 1. Initial pre-process for microbiome data
     
1. Define the raw ASVs and metadta files in the following format:
   1. ASVs file ("for_preprocess.csv"):
        - Each row represents a sample and each column represents an ASV.
        - The first column containing the samples Ids and named "ID".
        - The last row contains the taxonomy structure, and named "taxonomy".
        - Note: "for_preprocess.csv" is a file that contains the raw ASVs table in the required format. An example file can be found in "Example_Data" folder.
                   "folder" should be set as the directory path of the file.
        ```python
        preprocess_path  = f"{folder}/for_preprocess.csv"
        ```
        
   3. Metadata file ("metadata.csv"):
        - The first column containing the samples Ids and named "ID".
        - The file should contains at least on column representing a specific target.
        - Note: "metadata.csv" is a file that contains the metadata table in the require format. An example file can be found in "Example_Data" folder.
                       "folder" should be set as the directory path of the file.
                  
        ```python
        tag_path = f"{folder}/metadata.csv"
        ```
        
 3. Define output names path for saving output files:

     ```python
     MIPMLP_saving_name = f"MIPMLP_mean"
     preprocess_saving_name = f"for_preprocess"
     tag_saving_name = f"metadata"
     ```

 4. Apply pre-process of the data

    Parameters:
    - *preprocess_path (str):* Directory of the "preprocess" data CSV file
    - *tag_path (str):* Directory of the "metadata" CSV file
    - *folder (str):* Directory for saving output files
    - *MIPMLP_saving_name (str):* Name for saving "MIPMLP" relates DataFrames
    - *preprocess_saving_name (str):* Name for saving "preprocess" relates DataFrames
    - *tag_saving_name (str):* Name for saving "metadata" DataFrames
    - *normalization (str):* Normalization type for MIPMLP. Deafult is 'relative'.
    
    ```python
    apply_preprocess(preprocess_path, MIPMLP_path, tag_path, preprocess_saving_path, MIPMLP_saving_path, tag_saving_path, normalization='relative')
    ```
   - Note: normalization should be set as 'relative' for the Decomposition-Transformation Method reqiurments.
   
 4. Output

    The following files will be saved in the target directory:
    - MIPMLP_mean.csv": Processed file of the original "for_preprocess.csv", using 'relative' normalization.
    - for_preprocess_ordered.csv": "for_preprocess.csv" file after saving overlap ordered samples.
    - MIPMLP_mean_ordered.csv": "MIPMLP_mean.csv" file after saving overlap ordered samples.
    - metadata_ordered.csv": "metadata.csv" file after saving overlap ordered samples.
    - for_preprocess_filtered.csv": "for_preprocess_ordered.csv" file after saving samples passed quality check only.
    - MIPMLP_mean_filtered.csv": "MIPMLP_mean_ordered.csv" file after saving samples passed quality check only.
    - metadata_filtered.csv": "metadata_ordered.csv" file after saving samples passed quality check only.
