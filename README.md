# AGV Anomaly Detection using Isolation Forest and DBSCAN

This project focuses on detecting anomalies in **Automated Guided Vehicle (AGV)** operational data using **Isolation Forest** and **DBSCAN** algorithms. The goal is to identify unusual patterns in AGV behavior, such as irregular battery usage, erratic speeds, and extended stop times, which could indicate operational inefficiencies or malfunctions.

## Key Highlights
- **Objective:** Detect anomalies in AGV operational data using unsupervised learning techniques.
- **Dataset:** AGV dataset with features like battery voltage, speed, energy consumption, and stop times.
- **Preprocessing:**
  - Feature selection and removal of highly correlated features.
  - Feature engineering (e.g., battery voltage difference, energy consumption difference).
  - Standardization using `StandardScaler`.
- **Models:**
  - **DBSCAN:** Density-based clustering to identify anomalies as points in low-density regions.
  - **Isolation Forest:** Isolation-based anomaly detection to identify outliers through random splits.
- **Visualization:**
  - 2D and 3D PCA plots for visualizing clusters and anomalies.
  - Interactive plots using Plotly for better exploration of results.
- **Key Findings:**
  - **Anomalies Detected:** 4 anomalies were identified using Isolation Forest (contamination = 0.045).
  - **Key Anomalous Behaviors:** Higher battery voltage differences, extended stop times, and erratic speed variations.
  - **Comparison with DBSCAN:** High agreement between anomalies detected by both methods, with Isolation Forest identifying additional subtle anomalies.
- **Tools Used:** Python, Scikit-learn, Pandas, NumPy, Plotly, Flask.

## Future Improvements
- Experiment with other anomaly detection algorithms like **OPTICS** or **One-Class SVM**.
- Incorporate real-time monitoring for immediate anomaly detection.
- Explore advanced dimensionality reduction techniques like **t-SNE** or **UMAP** for better visualization.

## References
- [DBSCAN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
