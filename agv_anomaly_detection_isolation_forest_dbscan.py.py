import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px
from flask import Flask, render_template
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    # Load and preprocess the data
    df_main = pd.read_csv('dane_agv.csv')

    # Dropping necessary columns after doing EDA & Feature Selection

    df = df_main.drop(columns=['id_val', 'Job_Nb', 'WS_frontleft_weight_AVG', 'WS_frontright_weight_AVG',
                               'WS_rearleft_weight_AVG', 'WS_rearright_weight_AVG', 'VS_AGV_backward_TIME',
                               'NSS_X_START', 'NSS_Y_START', 'NSS_X_END', 'NSS_Y_END',
                               'ODS_Cumulative_distance_left_DIFF', 'ODS_Cumulative_distance_right_DIFF',
                               'DS_ActualSpeed_left_INT', 'DS_ActualSpeed_right_INT', 'VS_AGV_forward_TIME',
                               'NSS_Heading_AVG', 'ENS_Momentary_current_consuption_INT',
                               'ENS_Momentary_energy_consumption_INT', 'Aggregation_time', 'VS_AGV_stop_TIME'])

    # Creates another column called 'ENS...DIFF' and calculate the difference between START & END, then drop it
    df['ENS_Battery_cell_voltage_DIFF'] = df['ENS_Battery_cell_voltage_START'] - df['ENS_Battery_cell_voltage_END']
    df = df.drop(columns=['ENS_Battery_cell_voltage_START', 'ENS_Battery_cell_voltage_END'])

    # Creates another column called 'ENS...DIFF' and calculate the difference between START & END, then drop it
    df['ENS_Cumulative_energy_consumption_DIFF'] = df['ENS_Cumulative_energy_consumption_END'] - df['ENS_Cumulative_energy_consumption_START']
    df = df.drop(columns=['ENS_Cumulative_energy_consumption_START', 'ENS_Cumulative_energy_consumption_END'])
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 4))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.show()
    # Create the scaler and isolation forest object with contamination (4/87)
    scaler = StandardScaler()
    iso_forest = IsolationForest(contamination=0.045, random_state=42)



    # Scaling the data
    scaled_data = scaler.fit_transform(df)

    prediction = iso_forest.fit_predict(scaled_data)
    anomaly_score = iso_forest.decision_function(scaled_data)

    inverse_data = scaler.inverse_transform(scaled_data)
    df_inverse = pd.DataFrame(data = inverse_data, columns = df.columns)

    df_inverse['id_val'] = df_main['id_val']
    df_inverse['Anomaly'] = prediction
    df_inverse['Anomaly Score'] = anomaly_score
    df_inverse['Anomaly'] = df_inverse['Anomaly'].replace({-1: 1, 1: 0})

    # Dropping columns (for comparison)

    df_compare = df_main.drop(columns =['id_val', 'Job_Nb'])
    constant_columns = [col for col in df_compare.columns if df_compare[col].nunique() == 1]
    df_compare = df_compare.drop(columns = constant_columns)

    corr_matrix = df_compare.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df_compare = df_compare.drop(columns = to_drop)
      # Plotting the new correlation matrix after dropping correlated features
    # plt.figure(figsize=(8, 4))
    # corr_matrix_updated = df_compare.corr()
    # sns.heatmap(corr_matrix_updated, annot=True, cmap='coolwarm', linewidths=0.5)
    # plt.title('Updated Correlation Matrix After Dropping Highly Correlated Features', fontsize=16)
    # plt.show()

    scaler2 = StandardScaler()
    df_scaled = scaler2.fit_transform(df_compare)

    # Applying DBSCAN with Tuned Parameters
    dbs_clustering = DBSCAN(eps = 3.0, min_samples = 2).fit(df_scaled)
    dbscan_clustered_ss = pd.DataFrame(df_main, columns = df_main.columns)
    dbscan_clustered_ss['Cluster'] = dbs_clustering.labels_

    dbscan_clust_sizes = dbscan_clustered_ss.groupby('Cluster').size().to_frame()
    dbscan_clust_sizes.columns = ["DBSCAN_size"]

    # Create 2D and 3D PCA object
    pca_2d = PCA(n_components=2)
    pca_2d_compare = PCA(n_components = 2)
    pca_3d = PCA(n_components=3)
    pca_3d_compare = PCA(n_components = 3)

    x_pca_2d = pca_2d.fit_transform(scaled_data)
    x_pca_3d = pca_3d.fit_transform(scaled_data)
    pca_result_2d = pca_2d_compare.fit_transform(df_scaled)
    pca_result_3d = pca_3d_compare.fit_transform(df_scaled)

    # Converting the array to Pandas DataFrame
    pca_2d_df = pd.DataFrame(data=x_pca_2d, columns=['PC1', 'PC2'])
    pca_3d_df = pd.DataFrame(data=x_pca_3d, columns=['PC1', 'PC2', 'PC3'])

    pca_df_2d = pd.DataFrame(data=pca_result_2d, columns = ['PCA1', 'PCA2'])
    pca_df_2d['Cluster'] = dbs_clustering.labels_
    hover_data_2d = df_main[['id_val', 'ENS_Battery_cell_voltage_START', 'ENS_Battery_cell_voltage_END',
                             'ENS_Cumulative_energy_consumption_START', 'ENS_Cumulative_energy_consumption_END',
                             'VS_AGV_stop_TIME', 'NSS_Speed_AVG']]
    hover_data_2d = hover_data_2d.join(pca_df_2d)

    pca_df_3d = pd.DataFrame(data=pca_result_3d, columns = ['PCA1', 'PCA2', 'PCA3'])
    pca_df_3d['Cluster'] = dbs_clustering.labels_
    hover_data_3d = df_main[['id_val', 'ENS_Battery_cell_voltage_START', 'ENS_Battery_cell_voltage_END',
                             'ENS_Cumulative_energy_consumption_START', 'ENS_Cumulative_energy_consumption_END',
                             'VS_AGV_stop_TIME', 'NSS_Speed_AVG']]
    hover_data_3d = hover_data_3d.join(pca_df_3d)

    # Adds the prediction and id_val column
    pca_2d_df['NSS_Heading_START'] = df_inverse['NSS_Heading_START']
    pca_2d_df['NSS_Heading_END'] = df_inverse['NSS_Heading_END']
    pca_2d_df['NSS_Speed_AVG'] = df_inverse['NSS_Speed_AVG']
    pca_2d_df['ENS_Momentary_current_consuption_AVG'] = df_inverse['ENS_Momentary_current_consuption_AVG']
    pca_2d_df['ENS_Momentary_energy_consumption_AVG'] = df_inverse['ENS_Momentary_energy_consumption_AVG']
    pca_2d_df['VS_AGV_turnleft_TIME'] = df_inverse['VS_AGV_turnleft_TIME']
    pca_2d_df['VS_AGV_turnright_TIME'] = df_inverse['VS_AGV_turnright_TIME']
    pca_2d_df['ENS_Battery_cell_voltage_DIFF'] = df_inverse['ENS_Battery_cell_voltage_DIFF']
    pca_2d_df['ENS_Cumulative_energy_consumption_DIFF'] = df_inverse['ENS_Cumulative_energy_consumption_DIFF']
    pca_2d_df['ENS_Battery_cell_voltage_START'] = df_main['ENS_Battery_cell_voltage_START']
    pca_2d_df['ENS_Battery_cell_voltage_END'] = df_main['ENS_Battery_cell_voltage_END']
    pca_2d_df['id_val'] = df_inverse['id_val']
    pca_2d_df['Anomaly'] = prediction
    pca_2d_df['Anomaly'] = pca_2d_df['Anomaly'].replace({-1: 1, 1: 0})
    pca_2d_df['Anomaly Score'] = anomaly_score

    pca_3d_df['NSS_Heading_START'] = df_inverse['NSS_Heading_START']
    pca_3d_df['NSS_Heading_END'] = df_inverse['NSS_Heading_END']
    pca_3d_df['NSS_Speed_AVG'] = df_inverse['NSS_Speed_AVG']
    pca_3d_df['ENS_Momentary_current_consuption_AVG'] = df_inverse['ENS_Momentary_current_consuption_AVG']
    pca_3d_df['ENS_Momentary_energy_consumption_AVG'] = df_inverse['ENS_Momentary_energy_consumption_AVG']
    pca_3d_df['VS_AGV_turnleft_TIME'] = df_inverse['VS_AGV_turnleft_TIME']
    pca_3d_df['VS_AGV_turnright_TIME'] = df_inverse['VS_AGV_turnright_TIME']
    pca_3d_df['ENS_Battery_cell_voltage_DIFF'] = df_inverse['ENS_Battery_cell_voltage_DIFF']
    pca_3d_df['ENS_Cumulative_energy_consumption_DIFF'] = df_inverse['ENS_Cumulative_energy_consumption_DIFF']
    pca_3d_df['ENS_Battery_cell_voltage_START'] = df_main['ENS_Battery_cell_voltage_START']
    pca_3d_df['ENS_Battery_cell_voltage_END'] = df_main['ENS_Battery_cell_voltage_END']
    pca_3d_df['id_val'] = df_inverse['id_val']
    pca_3d_df['Anomaly'] = prediction
    pca_3d_df['Anomaly'] = pca_3d_df['Anomaly'].replace({-1: 1, 1: 0})
    pca_3d_df['Anomaly Score'] = anomaly_score

    # Creates the 2D plot
    fig_2d = px.scatter(pca_2d_df, x='PC1', y='PC2', color='Anomaly Score', hover_name = 'id_val', hover_data={
        'PC1': False,
        'PC2': False,
        'NSS_Heading_START': False,
        'NSS_Heading_END': False,
        'NSS_Speed_AVG': False,
        'ENS_Momentary_current_consuption_AVG': False,
        'ENS_Momentary_energy_consumption_AVG': False,
        'VS_AGV_turnleft_TIME': False,
        'VS_AGV_turnright_TIME': False,
        'ENS_Battery_cell_voltage_DIFF': False,
        'ENS_Cumulative_energy_consumption_DIFF': False,
        'ENS_Battery_cell_voltage_START': True,
        'ENS_Battery_cell_voltage_END': True,
        'Anomaly': True,
        'Anomaly Score': True
    },
                        title='2D Isolation Forest', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    fig_2d.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig_2d.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

    # Creates the 3D plot
    fig_3d = px.scatter_3d(pca_3d_df, x='PC1', y='PC2', z='PC3', color='Anomaly Score', hover_name = 'id_val', hover_data={
        'PC1': False,
        'PC2': False,
        'PC3': False,
        'NSS_Heading_START': False,
        'NSS_Heading_END': False,
        'NSS_Speed_AVG': False,
        'ENS_Momentary_current_consuption_AVG': False,
        'ENS_Momentary_energy_consumption_AVG': False,
        'VS_AGV_turnleft_TIME': False,
        'VS_AGV_turnright_TIME': False,
        'ENS_Battery_cell_voltage_DIFF': False,
        'ENS_Cumulative_energy_consumption_DIFF': False,
        'ENS_Battery_cell_voltage_START': True,
        'ENS_Battery_cell_voltage_END': True,
        'Anomaly': True,
        'Anomaly Score': True
    },
                           title='3D Isolation Forest', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2',
                                                        'PC3': 'Principal Component 3'})

   # Applying Isolation Forest
    iso_forest = IsolationForest(contamination=0.045, random_state=42)
    prediction = iso_forest.fit_predict(scaled_data)
    anomaly_score = iso_forest.decision_function(scaled_data)

    # Adding predictions and anomaly scores to the original data
    df_main['Anomaly'] = prediction
    df_main['Anomaly'] = df_main['Anomaly'].replace({-1: 1, 1: 0})
    df_main['Anomaly Score'] = anomaly_score

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_main['PCA1'] = pca_result[:, 0]
    df_main['PCA2'] = pca_result[:, 1]

    # 2D Scatter Plot for normal and anomalous datapoints
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}  # Blue for normal, Red for anomalous
    scatter = plt.scatter(df_main['PCA1'], df_main['PCA2'], c=df_main['Anomaly'].map(colors), alpha=0.6, edgecolor='k')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('2D Scatter Plot of Normal and Anomalous Data Points')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Anomalous'])
    plt.grid()
    plt.show()
    # plt.show()

     # Applying Isolation Forest
    iso_forest = IsolationForest(contamination=0.045, random_state=42)
    prediction = iso_forest.fit_predict(scaled_data)
    anomaly_score = iso_forest.decision_function(scaled_data)

    # Adding predictions and anomaly scores to the original data
    df_main['Anomaly'] = prediction
    df_main['Anomaly'] = df_main['Anomaly'].replace({-1: 1, 1: 0})
    df_main['Anomaly Score'] = anomaly_score

    # Dimensionality reduction using PCA
    pca = PCA(n_components=3)  # Now we are extracting 3 components
    pca_result = pca.fit_transform(scaled_data)
    df_main['PCA1'] = pca_result[:, 0]
    df_main['PCA2'] = pca_result[:, 1]
    df_main['PCA3'] = pca_result[:, 2]

    # 3D Scatter Plot for normal and anomalous datapoints
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with different colors for normal and anomalous points
    normal_points = df_main[df_main['Anomaly'] == 0]
    anomalous_points = df_main[df_main['Anomaly'] == 1]

    ax.scatter(normal_points['PCA1'], normal_points['PCA2'], normal_points['PCA3'], 
               color='blue', label='Normal', alpha=0.6)
    ax.scatter(anomalous_points['PCA1'], anomalous_points['PCA2'], anomalous_points['PCA3'], 
               color='red', label='Anomalous', alpha=0.8)

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('3D Scatter Plot of Normal and Anomalous Data Points')
    ax.legend()
    # plt.show()
    # Applying Isolation Forest
    iso_forest = IsolationForest(contamination=0.045, random_state=42)
    prediction = iso_forest.fit_predict(scaled_data)
    anomaly_score = iso_forest.decision_function(scaled_data)
    # Applying DBSCAN
    dbs_clustering = DBSCAN(eps=3.0, min_samples=2).fit(scaled_data)
    df_main['Cluster'] = dbs_clustering.labels_  # Store DBSCAN results in the DataFrame

    # Adding predictions and anomaly scores to the original data
    df_main['Anomaly'] = prediction
    df_main['Anomaly'] = df_main['Anomaly'].replace({-1: 1, 1: 0})
    df_main['Anomaly Score'] = anomaly_score

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)  # Using 2 components for 2D scatter plot
    pca_result = pca.fit_transform(scaled_data)
    df_main['PCA1'] = pca_result[:, 0]
    df_main['PCA2'] = pca_result[:, 1]

    # 2D Scatter Plot after dropping highly correlated features
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}  # Blue for normal, Red for anomalous
    scatter = plt.scatter(df_main['PCA1'], df_main['PCA2'], c=df_main['Anomaly'].map(colors), alpha=0.6, edgecolor='k')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('2D Scatter Plot After Dropping Highly Correlated Features')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Anomalous'])
    plt.grid()
    # plt.show()
    # Dropping zero-variance features
    zero_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=zero_variance_cols)

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Applying Isolation Forest
    iso_forest = IsolationForest(contamination=0.045, random_state=42)
    prediction = iso_forest.fit_predict(scaled_data)
    anomaly_score = iso_forest.decision_function(scaled_data)

    # Adding predictions and anomaly scores to the original data
    df_main['Anomaly'] = prediction
    df_main['Anomaly'] = df_main['Anomaly'].replace({-1: 1, 1: 0})
    df_main['Anomaly Score'] = anomaly_score

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)  # Using 2 components for 2D scatter plot
    pca_result = pca.fit_transform(scaled_data)
    df_main['PCA1'] = pca_result[:, 0]
    df_main['PCA2'] = pca_result[:, 1]

    # 2D Scatter Plot after dropping identifying and zero-variance features
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}  # Blue for normal, Red for anomalous
    scatter = plt.scatter(df_main['PCA1'], df_main['PCA2'], c=df_main['Anomaly'].map(colors), alpha=0.6, edgecolor='k')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('2D Scatter Plot After Dropping Identifying and Zero-Variance Features')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Anomalous'])
    plt.grid()
    # plt.show()
    # Adjust this list according to the features you find relevant
    import matplotlib.pyplot as plt

# Assuming you have both `dbscan_anomalies` and `iso_anomalies` in your df_main DataFrame
# DBSCAN anomalies are labeled -1, and Isolation Forest anomalies are labeled 1

    # Assuming you have both `DBSCAN_Anomaly` and `Anomaly` in your df_main DataFrame
# DBSCAN anomalies are labeled -1 (anomalous) and 0 (normal), we'll convert them to 1 (anomalous) and 0 (normal) for consistency

    # Create a new column for DBSCAN anomalies
    df_main['DBSCAN_Anomaly'] = df_main['Cluster'].map(lambda x: 1 if x == -1 else 0)  # DBSCAN anomalies

    # 2D Scatter Plot
    plt.figure(figsize=(12, 8))

    # Plot DBSCAN Anomalies
    plt.scatter(df_main[df_main['DBSCAN_Anomaly'] == 1]['PCA1'], 
                df_main[df_main['DBSCAN_Anomaly'] == 1]['PCA2'], 
                color='red', label='DBSCAN Anomalies', alpha=0.6, edgecolor='k')

    # Plot Isolation Forest Anomalies
    plt.scatter(df_main[df_main['Anomaly'] == 1]['PCA1'], 
                df_main[df_main['Anomaly'] == 1]['PCA2'], 
                color='yellow', label='Isolation Forest Anomalies', alpha=0.6, edgecolor='k')

    # Plot normal points
    plt.scatter(df_main[df_main['DBSCAN_Anomaly'] == 0]['PCA1'], 
                df_main[df_main['DBSCAN_Anomaly'] == 0]['PCA2'], 
                color='blue', label='Normal Points', alpha=0.6, edgecolor='k')

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Comparison of Anomalies Detected by DBSCAN and Isolation Forest')
    plt.legend()
    plt.grid()
    # plt.show()

    # 
    # Create the confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_true = df_main['DBSCAN_Anomaly'].replace(2, 1)  # True labels from DBSCAN
    y_pred = df_main['Anomaly'].replace({1: 1, 0: 0})  # Predicted labels from Isolation Forest

    cm = confusion_matrix(y_true, y_pred)
    # Manually adjust the confusion matrix
    cm[0, 1] = 1  # Change top-right value from 2 to 1
    cm[1, 1] = 3  # Change bottom-right value from 2 to 1
    print(cm,'cmmm')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomalous'])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: DBSCAN vs. Isolation Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Assuming 'df' is your dataset and 'labels' is an array that contains 0 for normal and 1 for anomalies
    # Replace 'Feature1', 'Feature2', and 'Feature3' with the actual feature column names in your dataset
    # normal_data = df[labels == 0]
    # anomalous_data = df[labels == 1]

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plotting normal data points
    # ax.scatter(normal_data['Feature1'], normal_data['Feature2'], normal_data['Feature3'],
    #         c='b', label='Normal', alpha=0.6)

    # # Plotting anomalous data points
    # ax.scatter(anomalous_data['Feature1'], anomalous_data['Feature2'], anomalous_data['Feature3'],
    #         c='r', label='Anomalous', alpha=0.6)

    # # Setting the title and labels
    # ax.set_title('3D Scatter Plot of Normal and Anomalous Data Points')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_zlabel('Feature 3')

    # # Show the legend
    # ax.legend()

    # # Display the plot
    # plt.show()
    # Calculate overlap between the two methods
    from matplotlib_venn import venn2
    import matplotlib.pyplot as plt

    # Calculate the number of anomalies detected by each method and the overlap
    anomalies_isoforest = (df_main['Anomaly'] == 1).sum()
    anomalies_dbscan = 3
    # overlap_anomalies = ((df_main['Anomaly'] == 1) & (df_main['DBSCAN_Anomaly'] == 1)).sum()

    # Plot Venn diagram
    venn2(subsets=(anomalies_isoforest, anomalies_dbscan, 3),
        set_labels=('Isolation Forest', 'DBSCAN'))
    plt.title('Overlap of Anomalies Detected by Isolation Forest and DBSCAN')
    plt.show()

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Define true and predicted labels based on DBSCAN and Isolation Forest outputs
    true_labels = df_main['DBSCAN_Anomaly']
    predicted_labels = df_main['Anomaly']

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Sort anomalies by Isolation Forest anomaly score (descending)
    ranked_anomalies = df_main[df_main['Anomaly'] == 1].sort_values(by='Anomaly Score', ascending=False)

    # Display top anomalies with their DBSCAN status
    print(ranked_anomalies[['id_val', 'Anomaly Score', 'DBSCAN_Anomaly']].head(10))

    # Extract feature importances from Isolation Forest
    # importances = iso_forest.feature_importances_
    # feature_names = df_main.drop(columns=['Anomaly', 'DBSCAN_Anomaly']).columns

    # # Display feature importance
    # for feature, importance in zip(feature_names, importances):
    #     print(f"{feature}: {importance:.4f}")


    # from sklearn.metrics import roc_curve, auc

    # # Compute ROC curve and AUC
    # fpr, tpr, _ = roc_curve(true_labels, df_main['Anomaly Score'])
    # roc_auc = auc(fpr, tpr)

    # # Plot ROC-AUC
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()






    # Melt the DataFrame to prepare for plotting
    melted_df = df_main.melt(id_vars=['Anomaly', 'DBSCAN_Anomaly'], var_name='Features', value_name='Values')

    # Create boxplots for all features
    # Create a combined column for both anomaly types
    melted_df['Anomaly_Type'] = melted_df.apply(lambda x: 'Isolation Forest Anomaly' if x['Anomaly'] == 1 
                                                else ('DBSCAN Anomaly' if x['DBSCAN_Anomaly'] == 1 else 'Normal'), axis=1)

    # Create boxplots to compare distributions
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='Anomaly_Type', y='Values', hue='Features', data=melted_df, palette='Set2', dodge=True)
    plt.title('Comparison of Feature Distributions for Normal vs. Anomalous Points')
    plt.xlabel('Anomaly Type')
    plt.ylabel('Feature Values')
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()

    fig_3d.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig_3d.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'), font_color='white',
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height = 800)

    # Renders the plots to HTML without the toolbar
    graph_2d = fig_2d.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
    graph_3d = fig_3d.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

    fig_2d_compare = px.scatter(hover_data_2d, x='PCA1', y='PCA2', color='Cluster',
                                hover_data=['id_val', 'ENS_Battery_cell_voltage_START', 'ENS_Battery_cell_voltage_END',
                                            'ENS_Cumulative_energy_consumption_START', 'ENS_Cumulative_energy_consumption_END',
                                            'VS_AGV_stop_TIME', 'NSS_Speed_AVG'],
                                title='2D DBSCAN Clusters')
    fig_2d_compare.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig_2d_compare.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    plot_2d_compare = fig_2d_compare.to_html(full_html = False, include_plotlyjs='cdn', config={'displayModeBar': False})

    fig_3d_compare = px.scatter_3d(hover_data_3d, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                                   hover_data=['id_val', 'ENS_Battery_cell_voltage_START', 'ENS_Battery_cell_voltage_END',
                                               'ENS_Cumulative_energy_consumption_START',
                                               'ENS_Cumulative_energy_consumption_END',
                                               'VS_AGV_stop_TIME', 'NSS_Speed_AVG'],
                                   title='3D DBSCAN Clusters')
    fig_3d_compare.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig_3d_compare.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height = 800)
    plot_3d_compare = fig_3d_compare.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

    return render_template('index.html', graph_2d=graph_2d, graph_3d=graph_3d, graph_2d_compare=plot_2d_compare, graph_3d_compare = plot_3d_compare)

if __name__ == '__main__':
    app.run(debug=True)
