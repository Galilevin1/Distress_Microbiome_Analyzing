### Integrating Gut Microbiota and Violence Exposure Metrics to Classify Psychological Distress in Middle-Aged Adults

Analyses the relation between microbiome compositions and host violence exposure level, to its psychological distress level.

## Analysis
1. "Statistical_Analysis.ipynb"- Statistical visualization and analyses applied in google colab
2. "Preprocess_microbiome.py" - A PyCharm script to preprocess the microbiome file, filter quality samples and microbes and use the MIPMLP library to merge similar features based on the taxonomy, scale distributions, standardize to z-scores, and perform dimension reduction at the species level.
3. "Models_Analyses.py" - performing models to invastigate the relation of microbiome, violence metrics, demographic feaures, and psychological distress.

## Apply methods:

## 1. Filter quality microbiome data and use MIPMLP pre-process ("Preprocess_microbiome.py")
     
1. Define the raw ASVs and metadta files in the following format:
   1. ASVs file ("for_preprocess.csv"):
        - Each row represents a sample and each column represents an ASV.
        - The first column containing the samples Ids and named "ID".
        - The last row contains the taxonomy structure, and named "taxonomy".
        - Note: "for_preprocess.csv" is a file that contains the raw ASVs table in the required format.
                   "folder" should be set as the directory path of the file.
        ```python
        preprocess_path  = f"{folder}/for_preprocess.csv"
        ```
        
   3. Metadata file ("metadata.csv"):
        - The first column containing the samples Ids and named "ID".
        - The file should contains the target column: Psychological distress, along with the violence features: "VIO1", "VIO2", "VIO3, "VIO4", "VIO5", "viol_total",
           and demographic features: Age, bmi, sex.
        - Note: "metadata.csv" is a file that contains the metadata table in the require format. 
                       "folder" should be set as the directory path of the file.
                  
        ```python
        tag_path = f"{folder}/metadata.csv"
        ```
        
 3. Define output names path for saving output files:

     ```python
     MIPMLP_saving_name = f"MIPMLP_scaled_adj"
     preprocess_saving_name = f"for_preprocess"
     tag_saving_name = f"metadata_adj"
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
    - MIPMLP_scaled_adj.csv": Processed file of the original "for_preprocess.csv", using 'relative' normalization.
    - for_preprocess_ordered.csv": "for_preprocess.csv" file after saving overlap ordered samples.
    - MIPMLP_scaled_adj_ordered.csv": "MIPMLP_mean.csv" file after saving overlap ordered samples.
    - metadata_adj_ordered.csv": "metadata.csv" file after saving overlap ordered samples.
    - for_preprocess_filtered.csv": "for_preprocess_ordered.csv" file after saving samples passed quality check only.
    - MIPMLP_mean_filtered.csv": "MIPMLP_mean_ordered.csv" file after saving samples passed quality check only.
    - metadata_filtered.csv": "metadata_ordered.csv" file after saving samples passed quality check only.
