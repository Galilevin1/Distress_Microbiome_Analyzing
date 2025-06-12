import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as st
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_predict, cross_validate, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statsmodels.stats.multicomp as mc
import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import shap
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
from scipy.stats import f_oneway

# Extract microbiome from literature
def Extract_mirobes_names(MIPMLP, Name_list):
    microbes_list = []
    for name in Name_list:
        microbes_list += [col for col in MIPMLP.columns if name in col]
    return list(set(microbes_list))

def model_microbiome_features(Data, target, model, condition_name, iter_num=1, param_grid=None, preprocess=None, tune_params=False, params_filename='best_params.json'):
    def save_best_params(best_params_dict, filename):
        with open(filename, 'w') as file:
            json.dump(best_params_dict, file)
        print(f"Best parameters saved to {filename}")

    def load_best_params(filename):
        try:
            with open(filename, 'r') as file:
                best_params_dict = json.load(file)
            print(f"Best parameters loaded from {filename}")
        except FileNotFoundError:
            best_params_dict = {}
            print(f"No file found. Starting with an empty dictionary.")
        return best_params_dict

    print(f"{condition_name}:")
    AUC_list = []
    best_params_dict = load_best_params(params_filename)

    # Determine the minimum class count
    min_class_count = np.min(np.bincount(target))
    n_splits = min(5, min_class_count)  # Use 5 splits instead of 10 to reduce runtime

    if tune_params:
        # Split to train and test sets
        X_train, X_test, y_train, y_test = train_test_split(Data, target, test_size=0.2, stratify=target, random_state=42)
        # Apply Preprocessing
        if preprocess:
            X_train_transformed = preprocess.fit_transform(X_train)
            X_test_transformed = preprocess.transform(X_test)
        else:
            X_train_transformed = X_train
            X_test_transformed = X_test
        # Hyperparameter tuning
        if param_grid is not None:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                                       scoring='roc_auc', n_jobs=-1)  # Use all available cores
            grid_search.fit(X_train_transformed, y_train)
            best_params = grid_search.best_params_
            model.set_params(**best_params)
            if best_params_dict is not None:
                best_params_dict[condition_name] = best_params
            print(f"Best parameters found: {best_params}")

        # Train the model with the best parameters
        model.fit(X_train_transformed, y_train, early_stopping_rounds=10, eval_set=[(X_test_transformed, y_test)], verbose=False)

        # Predictions
        y_pred_proba_test = model.predict_proba(X_test_transformed)[:, 1]

        # Calculate ROC AUC for the test set
        roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
        AUC_list.append(roc_auc_test)
        # Save the best parameters
        save_best_params(best_params_dict, params_filename)

    else:
        if condition_name in best_params_dict:
            model.set_params(**best_params_dict[condition_name])

        for i_iter in range(iter_num):
            # Split to train and test sets
            X_train, X_test, y_train, y_test = train_test_split(Data, target, test_size=0.2, stratify=target, random_state=i_iter)
            # Apply Preprocessing
            if preprocess:
                X_train_transformed = preprocess.fit_transform(X_train)
                X_test_transformed = preprocess.transform(X_test)
            else:
                X_train_transformed = X_train
                X_test_transformed = X_test
            # # Cross-validation
            # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # y_pred_cv = cross_val_predict(model, X_train_transformed, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

            # Train the model
            model.fit(X_train_transformed, y_train, early_stopping_rounds=10, eval_set=[(X_test_transformed, y_test)], verbose=False)

            # Predictions
            y_pred_proba_test = model.predict_proba(X_test_transformed)[:, 1]

            # Calculate ROC AUC for cross-validation and test set
            roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
            AUC_list.append(roc_auc_test)

    return AUC_list, model, X_train_transformed


# AUC distribution
def AUC_distribution(AUC_list):
    sns.histplot(AUC_list, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUC Values')
    plt.show()

# Plot ROC curve
def ROC_CURVE_Visualization(fpr, tpr, roc_auc, condition= "test"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve- {condition}')
    plt.legend(loc='lower right')
    plt.show()

# Confusion Matrix
def Confusion_matrix(y_test, y_pred_test):
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Distress', 'Distress'], yticklabels=['No Distress', 'Distress'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def BoxPlots(AUC_list, headlines_list, location_list):
    # Create a boxplot
    plt.figure(figsize=(10, 6))
    boxprops = dict(color='black', facecolor=(197/255, 224/255, 180/255))
    capprops = dict(color='black')
    whiskerprops = dict(color='black')
    flierprops = dict(markerfacecolor='black', marker='o')
    medianprops = dict(color='black')
    plt.boxplot(AUC_list, vert=True, patch_artist=True,
                boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops,
                flierprops=flierprops, medianprops=medianprops)  #showmeans=True,
    # Set labels
    plt.xticks(location_list, headlines_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('AUC Values', fontsize=22)
    plt.show()

def Histograms(AUC_list, headlines_list):
    num_plots = len(AUC_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), sharey=True)

    for i, ax in enumerate(axes):
        ax.hist(AUC_list[i], bins=10, color=(197/255, 224/255, 180/255), edgecolor='black')
        ax.set_title(headlines_list[i], fontsize=14)
        ax.set_xlabel('AUC', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.show()

def return_leaf_names(bacteria_names):

    # Identify non-leaf taxa (those that are prefixes of longer taxa)
    non_leaf_taxa = set()
    for name in bacteria_names:
        parts = name.split(";")
        for i in range(1, len(parts)):
            parent = ";".join(parts[:i])
            non_leaf_taxa.add(parent)

    # Filter out non-leaf names (i.e. keep only real leaves)
    leaf_names = [name for name in bacteria_names if name not in non_leaf_taxa]

    return leaf_names

def rename_microbes(features):
    renamed_features = []
    for feature in features:
        parts = feature.split(';')
        Class = next((part.split('__')[1] for part in parts if part.startswith('c__')), '')
        order = next((part.split('__')[1] for part in parts if part.startswith('o__')), '')
        family = next((part.split('__')[1] for part in parts if part.startswith('f__')), '')
        genus = next((part.split('__')[1] for part in parts if part.startswith('g__')), '')
        species = next((part.split('__')[1] for part in parts if part.startswith('s__')), '')
        if species != '' and genus != '':
            renamed_features.append(f"{genus} (g), {species} (s)")
        elif species == '' and genus != '':
            renamed_features.append(f"{genus} (g), __(s)")
        elif genus == '' and family != '':
            renamed_features.append(f"{family} (f), __(g), __(s)")
        elif family == '' and order != '':
            renamed_features.append(f"{order} (o), __(f), __(g), __(s)")
        elif order == '':
            renamed_features.append(f"{Class} (c), __(o), __(f), __(g), __(s)")
    return renamed_features

def Feature_importance_shap(model, X_train):
    # SHAP feature importance
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    # Extract the most important features
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': shap_importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    # Get the top 10 most important features
    top_10_features = list(set(importance_df['feature'].head(10).values))
    top_10_renamed = rename_microbes(top_10_features)
    # Get the top 10 most important features
    top_20_features = list(set(importance_df['feature'].head(20).values))
    top_20_renamed = rename_microbes(top_20_features)
    # Filter SHAP values for the top 10 features
    top_10_indices = [X_train.columns.get_loc(feature) for feature in top_10_features]
    shap_values_top_10 = shap_values[:, top_10_indices]
    # Create a new SHAP values object with the renamed features
    shap_values_top_10_renamed = shap.Explanation(
        values=shap_values_top_10.values,
        base_values=shap_values_top_10.base_values,
        data=shap_values_top_10.data,
        feature_names=top_10_renamed
    )
    # Change the font size and family for the plots
    plt.rcParams.update({
        'font.size': 18,
        #'font.family': 'Calibri Light (Headings)'
    })
    # Summary plot - bar
    #shap.summary_plot(shap_values_top_10_renamed, plot_type="bar", max_display=10)
    # plt.xlabel('Mean SHAP value (impact on model output)', fontsize=18, fontfamily='Calibri Light (Headings)')
    # plt.xticks(fontsize=22, fontfamily='Calibri Light (Headings)')
    # plt.yticks(fontsize=22, fontfamily='Calibri Light (Headings)')
    # plt.title('SHAP Summary Plot (Top 10 Features)', fontsize=20, fontfamily='Calibri Light (Headings)')
    # plt.tight_layout()
    # plt.show()
    # Detailed SHAP values plot
    shap.summary_plot(shap_values_top_10_renamed, max_display=10)
    # plt.xlabel('SHAP value (impact on model output)', fontsize=18, fontfamily='Calibri Light (Headings)')
    # plt.xticks(fontsize=22, fontfamily='Calibri Light (Headings)')
    # plt.yticks(fontsize=22, fontfamily='Calibri Light (Headings)')
    # plt.title('SHAP Values Plot (Top 10 Features)', fontsize=20, fontfamily='Calibri Light (Headings)')
    # plt.tight_layout()
    # plt.show()
    # Print the top 10 most important features
    print("Top 10 most important features:")
    print(top_20_renamed)
    return shap_values, top_10_features, top_20_features, top_10_renamed, top_20_renamed

def Feature_importance_XGboost(model, X_train):
    # Get feature importance from XGBoost model
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    # Get the top 10 most important features
    top_10_features = list(importance_df['feature'].head(10).values)
    top_10_scores = list(importance_df['importance'].head(10).values)
    top_10_renamed = rename_microbes(top_10_features)
    # Get the top 20 most important features
    top_20_features = list(importance_df['feature'].head(20).values)
    top_20_scores = list(importance_df['importance'].head(20).values)
    top_20_renamed = rename_microbes(top_20_features)
    # Plot the top 10 most important features
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_renamed, top_10_scores)
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
    # Print the top 10 most important features
    print("Top 10 most important features:")
    for feature, score in zip(top_10_renamed, top_10_scores):
        print(f"{feature}: {score:.4f}")
    # Print the top 20 most important features
    print("\nTop 20 most important features:")
    for feature, score in zip(top_20_renamed, top_20_scores):
        print(f"{feature}: {score:.4f}")
    return top_10_features, top_10_scores, top_10_renamed, top_20_features, top_20_scores, top_20_renamed

def plot_cov_microbe_shap(top_microbes_distress, top_microbes_violence, shap_values_distress, shap_values_violence,
                          X_train_distress, X_train_violence, top_microbes_renamed_distress, top_microbes_renamed_vio):
    # Combine top microbes
    combined_top_microbes = set(top_microbes_distress).union(set(top_microbes_violence))
    combined_top_microbes = list(combined_top_microbes)
    combined_top_names = set(top_microbes_renamed_distress).union(set(top_microbes_renamed_vio))
    combined_top_names = list(combined_top_names)
    #renamed_combined_top_microbes = rename_microbes(combined_top_microbes)

    # Initialize data for the combined bar plot
    combined_importance = []

    for microbe, renamed_microbe in zip(combined_top_microbes, combined_top_names):  # ip(combined_top_microbes, renamed_combined_top_microbes)
        distress_cov = np.cov(shap_values_distress.values[:, X_train_distress.columns.get_loc(microbe)],
                              X_train_distress[microbe].values)[0, 1] if microbe in top_microbes_distress else 0
        violence_cov = np.cov(shap_values_violence.values[:, X_train_violence.columns.get_loc(microbe)],
                              X_train_violence[microbe].values)[0, 1] if microbe in top_microbes_violence else 0

        combined_importance.append({'microbe': renamed_microbe,
                                    'Cov_Psychological_Distress': distress_cov,
                                    'Cov_Violence_Total': violence_cov})

    combined_importance_df = pd.DataFrame(combined_importance)

    # Plot the combined bar plot rotated by 90 degrees
    combined_importance_df.set_index('microbe')[['Cov_Psychological_Distress', 'Cov_Violence_Total']].plot(kind='barh', stacked=True,
                                                                                       figsize=(10, 6))
    plt.ylabel('Microbe')
    plt.xlabel('Covariance Value')
    plt.title('Combined Covariance Values for Distress and Violence Targets')
    plt.legend(['Psychological Distress Target', 'Violence Total Target'])
    plt.show()

def plot_cov_microbe_feature_importance_XGboost(X_train_distress, X_train_violence, top_microbes_distress, top_microbes_violence,
                                                top_scores_distress, top_scores_violence, top_microbes_renamed_distress, top_microbes_renamed_vio):
    # Combine top microbes
    combined_top_microbes = set(top_microbes_distress).union(set(top_microbes_violence))
    combined_top_microbes = list(combined_top_microbes)
    combined_top_names = set(top_microbes_renamed_distress).union(set(top_microbes_renamed_vio))
    combined_top_names = list(combined_top_names)

    # Initialize data for the combined bar plot
    combined_importance = []

    for microbe, renamed_microbe in zip(combined_top_microbes, combined_top_names):
        distress_cov = np.cov(top_scores_distress[top_microbes_distress == microbe], X_train_distress[microbe].values)[0, 1] if microbe in top_microbes_distress else 0
        violence_cov = np.cov(top_scores_violence[top_microbes_violence == microbe], X_train_violence[microbe].values)[0, 1] if microbe in top_microbes_violence else 0

        combined_importance.append({'microbe': renamed_microbe,
                                    'Cov_Psychological_Distress': distress_cov,
                                    'Cov_Violence_Total': violence_cov})

    combined_importance_df = pd.DataFrame(combined_importance)

    # Plot the combined bar plot rotated by 90 degrees
    combined_importance_df.set_index('microbe')[['Cov_Psychological_Distress', 'Cov_Violence_Total']].plot(
        kind='barh', stacked=True,
        figsize=(10, 6))
    plt.ylabel('Microbe')
    plt.xlabel('Covariance Value')
    plt.title('Combined Covariance Values for Distress and Violence Targets')
    plt.legend(['Psychological Distress Target', 'Violence Total Target'])
    plt.show()


def apply_analsis(df_process, df_tag, features, target_distress, iter_num=100, tune_params=False, params_filename=None,
                  flag_clean=False, preprocessor=None):
    # Adjust Microbiome and Target data

    if flag_clean:

        # Drop rows with NaN values in features columns
        for col in features:
            df_tag[col] = df_tag[col].apply(lambda x: float(x) if x != ' ' else np.nan)
        df_tag.dropna(subset=features, inplace=True)
        # Drop rows with NaN values in target column
        df_tag[target_distress] = df_tag[target_distress].apply(lambda x: float(x) if x != ' ' else np.nan)
        df_tag.dropna(subset=target_distress, inplace=True)
        # Drop corresponding rows in data
        df_process = df_process.loc[df_tag.index]

    # Filter dfs to contain only leaf
    leaf_cols = return_leaf_names(df_process.columns)
    df_process = df_process[leaf_cols]

    # Handle Data and targets
    microbe_total_data = df_process.iloc[:, 1:]
    feature_data = df_tag[features]
    target_distress_values = df_tag[target_distress].apply(lambda x: 0 if x < 1.833 else 1)
    microbe_total_data.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in
                                  microbe_total_data.columns]

    # Model param grid
    param_grid = {
        'n_estimators': [50, 60, 70, 80, 90, 100, 120, 130, 150, 160, 170, 180, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Initialize model
    model = xgb.XGBClassifier(use_label_encoder=False)

    # Define literature known microbes
    Literature_microbes = ['Lactobacillus', 'plantarum', 'Parabacteroides', 'goldsteinii', 'Barnesiella',
                           'intestinihominis', 'Paraprevotella', 'Eubacterium', 'eligens', 'distasonis', 'Akkermansia',
                           'muciniphila', 'Bacteroides', 'massiliensis', 'Bifidobacterium', 'longum', 'Dialister',
                           'invisus', 'Roseburia', 'inulinivorans', 'Streptococcus', 'Acidameinococcus',
                           'Pilipacter', 'Lutispora', 'Proteiniclasticum', 'Thermodesulfobium', 'Ruminococcus',
                           'Anaerostipes', 'Clostridium', 'XIVa', 'Eisenbergiella', 'Lachnospira',
                           'Pseudoflavonifractor',
                           'Subdoligranulum', 'ramosum', 'Collinsella', 'Veillonella', 'Bifidocaterium', 'Gemmiger',
                           'Lactococcus', 'SMB53', 'Firmicutes', 'Tenericutes', 'coprostanoligenes', 'Ruminococcaceae'
                                                                                                     'UCG-014',
                           'Prevotella', 'Shigella', 'Escherichia', 'Fusobacterium', 'gnavus']

    # Getting Literature microbes names overlapping with the data
    microbe_literature_overlap_list = Extract_mirobes_names(microbe_total_data, list(set(Literature_microbes)))

    # Performing models

    # Total microbiome- Target: Distress
    AUC_list_all_microbes, model_all_microbes, X_train_all_microbes = model_microbiome_features(microbe_total_data,
                                                                                                target_distress_values,
                                                                                                model, "All_microbes",
                                                                                                iter_num, param_grid,
                                                                                                None, tune_params,
                                                                                                params_filename)
    # Violence and general Features- Targe: Distress
    AUC_list_features, model_features, X_train_features = model_microbiome_features(feature_data,
                                                                                    target_distress_values, model,
                                                                                    "Features",
                                                                                    iter_num, param_grid, preprocessor,
                                                                                    tune_params, params_filename)
    # Literature microbiome- Target: Distress
    AUC_list_literature_microbes, model_literature_microbes, X_train_literature_microbes = model_microbiome_features(
        microbe_total_data[microbe_literature_overlap_list], target_distress_values, model,
        "literature_microbes- Distress",
        iter_num, param_grid, None, tune_params, params_filename)
    # All microbiome SHAP- Target: Distress
    shap_values_distress, shap_top_10_microbes_distress, shap_top_20_microbes_distress, shap_top_10_renamed_distress, shap_top_20_renamed_distress = Feature_importance_shap(
        model_all_microbes, X_train_all_microbes)
    # top_10_microbes_distress, top_10_scores_distress, top_10_renamed_distress, top_20_microbes_distress, top_20_scores_distress, top_20_renamed_distress = Feature_importance_XGboost(model_all_microbes, X_train_all_microbes)
    AUC_list_literature_microbes_SHAP, model_literature_microbes_SHAP, X_train_literature_microbes_SHAP = model_microbiome_features(
        microbe_total_data[shap_top_10_microbes_distress], target_distress_values, model, "All_microbes SHAP- Distress",
        iter_num, param_grid, None, tune_params, params_filename)

    # Combine Violence and general features and Microbiome- Targe: Distress
    combined_data = pd.concat([microbe_total_data[shap_top_10_microbes_distress], feature_data], axis=1)
    combined_data.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in combined_data.columns]
    AUC_list_combine, model_combine, X_train_combine = model_microbiome_features(combined_data, target_distress_values,
                                                                                 model, "Combined data",
                                                                                 iter_num, param_grid, preprocessor,
                                                                                 tune_params, params_filename)
    # # Significant microbes
    # AUC_list_significant_microbes, model_significant_microbes, X_train_significant_microbes = model_microbiome_features(microbe_total_data[significant_microbes], target_distress_values, model, "Significant microbes",
    # iter_num, param_grid, preprocess, tune_params, params_filename)

    # print(len(microbe_literature_overlap_data))
    # print(len(df_process.columns))

    # Creating Box-Plots for the AUC results of all conditions
    AUC_list = [AUC_list_all_microbes, AUC_list_features, AUC_list_literature_microbes,
                AUC_list_literature_microbes_SHAP, AUC_list_combine]  # AUC_list_significant_microbes,
    headlines_list = ['All Microbes', 'Features', 'Literature\n Microbes', 'SHAP\n Microbes',
                      'Combine:\n Features & SHAP']  # 'Significant Microbes',
    for i in range(len(AUC_list)):
        aucs = AUC_list[i]
        median = np.median(aucs)
        std_median = np.sqrt(np.mean((aucs - median) ** 2))
        mean = np.mean(aucs)
        std = np.std(aucs)
        q1 = np.percentile(aucs, 25)
        q3 = np.percentile(aucs, 75)
        iqr = q3 - q1

        print(f"{headlines_list[i]}")
        print(f"Median: {median}, std median: {std_median} | Mean: {mean}, std: {std} | IQR: {iqr}")
        print("\n")
    AUC_distribution_results = BoxPlots(AUC_list, headlines_list, [1, 2, 3, 4, 5])
    Histograms(AUC_list, headlines_list)

    # Anova test between AUC lists
    auc_data = {
        'All Microbes': AUC_list_all_microbes,
        'Features': AUC_list_features,
        'Literature Microbes': AUC_list_literature_microbes,
        'SHAP Microbes': AUC_list_literature_microbes_SHAP,
        'Combine: Features & SHAP': AUC_list_combine
    }

    # Convert the data to a DataFrame
    df = pd.DataFrame(auc_data)

    # Perform ANOVA test
    f_value, p_value = stats.f_oneway(df['All Microbes'], df['Features'], df['Literature Microbes'],
                                      df['SHAP Microbes'], df['Combine: Features & SHAP'])

    print(f"ANOVA test: F-value = {f_value}, p-value = {p_value}")

    # Check if the ANOVA test is significant
    alpha = 0.05
    if p_value < alpha:
        print("ANOVA test is significant. Performing post hoc tests.")

        # Find the model with the highest mean AUC
        mean_auc_values = df.mean()
        best_model = mean_auc_values.idxmax()
        print(f"Best model: {best_model}")

        # Prepare data for post hoc tests
        data_melted = df.melt(var_name='model', value_name='auc')

        # Perform Dunn's post hoc test
        dunn_result = sp.posthoc_dunn(data_melted, val_col='auc', group_col='model', p_adjust='fdr_bh')

        # Extract comparisons involving the best model
        comparisons = dunn_result.loc[best_model]

        # Prepare the summary table
        summary_data = []
        for model, p_adj in comparisons.items():
            if model != best_model:
                meandiff = df[best_model].mean() - df[model].mean()
                lower = meandiff - 1.96 * df[best_model].std() / len(df[best_model]) ** 0.5  # 95% CI lower bound
                upper = meandiff + 1.96 * df[best_model].std() / len(df[best_model]) ** 0.5  # 95% CI upper bound
                reject = p_adj < alpha
                summary_data.append({
                    'group1': best_model,
                    'group2': model,
                    'meandiff': meandiff,
                    'p-adj': p_adj,
                    'lower': lower,
                    'upper': upper,
                    'reject': reject
                })

        summary_table = pd.DataFrame(summary_data)
        print(summary_table.to_string(index=False))

    else:
        print("ANOVA test is not significant. No post hoc tests performed.")


######################################################################################################################

# Example use

# Read data
project = "Distress_new"
df_process = pd.read_csv(f"Datas/{project}/1/MIPMLP_scaled_adj.csv")
df_tag = pd.read_csv(f"Datas/{project}/1/metadata_adj.csv")

# Define model parameters
target_distress = 'Psych_Dist'
features = ['VIO1', 'VIO2', 'VIO3', 'VIO4', 'VIO5', 'viol_total', 'Age', 'bmi']
categorical_columns = []
binary_columns = ['SEX_0M']
numeric_columns = ['Age', 'bmi']
preprocessor = ColumnTransformer(
               transformers=[
               ('numeric', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_columns),
               ('categorical', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder())]), categorical_columns),
               ('binary', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())]), binary_columns)],
               remainder= 'passthrough')
params_filename = f"Datas/{project}/1/best_params_1.json"

apply_analsis(df_process, df_tag, features, target_distress, iter_num=100, tune_params=False, params_filename=f"Datas/{project}/1/best_params_1.json",
                  flag_clean=False, preprocessor=None)


######################################################################################

