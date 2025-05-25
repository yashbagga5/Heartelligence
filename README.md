# Heartelligence
This project focuses on predicting hospital readmissions for heart failure patients and providing visual health insights through an interactive dashboard. It combines clinical data processing, machine learning modeling, and interactive health analytics using Python libraries and Streamlit.


🎯 Objectives
- Filter and extract heart failure cases from MIMIC-III.
- Predict 30-day readmission risk using machine learning.
- Visualize key patient metrics interactively.
- Empower patients with journaling and lifestyle tracking tools.
- Provide emergency response education for at-risk individuals.


📊 Features


- MIMIC-III dataset integration📁 
- Data preprocessing and heart failure diagnosis filtering⚙️ 
- Machine learning prediction of readmission risk (using XGBoost and RandomForest)🤖 
- Health trends visualization📈 
- Risk score calculator🧠 
- Correlation and anomaly detection📉 
- Health journaling and lifestyle tracking🧾 
- Emergency preparedness guidance🚨 
- Summary and performance reports📋 

🖼️ UI Preview



🧪 Technologies Used

- Pandas / NumPy:	Data loading and preprocessing
- Scikit-learn:	Model training, evaluation, metrics
- XGBoost:	Predictive modeling (classification)
- Streamlit:	Web-based interactive UI
- Plotly:	Rich, interactive visualizations
- MIMIC-III Dataset:	Real-world medical data (ADMISSIONS.csv, DIAGNOSES_ICD.csv, PATIENTS.csv)


📚 Explanation:

- pandas as pd: Used for data manipulation and analysis (especially with tables called DataFrames).
- numpy as np: Used for numerical operations, especially with arrays.
- datetime, timedelta: Used for handling and manipulating dates and times.
- train_test_split: Function from sklearn to split your dataset into training and test sets.
- classification_report: Generates precision, recall, F1-score, and support metrics.
- XGBClassifier: A high-performance machine learning model (gradient boosting) from XGBoost, used for classification tasks.


Machine Learning Approach

📁 Data Preparation (Notebook)

- Load CSVs from the MIMIC-III dataset
- Filter diagnosis codes for heart failure (ICD-9 codes)
- Merge relevant datasets: Diagnoses + Admissions + Patients
- Compute readmission risk based on features
- Train-test split and model evaluation

🤖 Model

- XGBoostClassifier in Jupyter Notebook
- RandomForestClassifier in Streamlit dashboard
- Evaluation metrics: accuracy, ROC-AUC, classification report

🌐 Dashboard Sections

- Overview:	Displays heart rate trends, averages, and risk indicators
- Data Analysis:	Summary statistics, time series, anomaly detection
- Predictions:	Risk prediction form and model performance
- Emergency Measures:	Instructions and preparedness guide for emergencies
- Health Journal:	Personal wellness tracking (mood, sleep, exercise)
- Lifestyle Guide:	Nutrition, exercise, and stress management tools
- About:	Tech stack and team information


🏗️ Project Architecture

1. Backend (Data Processing & Modeling – Jupyter Notebook)
   
  a)Data Sources: CSV files from the MIMIC-III clinical database:
  
    ADMISSIONS.csv (admission data)
    DIAGNOSES_ICD.csv (diagnosis codes)
    PATIENTS.csv (demographics)

  b)Preprocessing Steps:
  
    Filtered diagnosis codes related to heart failure (ICD-9).
    Merged patient, diagnosis, and admission data.
    Calculated time between discharges and next admissions.
    Labeled readmissions within 30 days (readmit_30d).
    Performed feature engineering (e.g., age, gender).

  c)Model Training:
  
    Split the data into training and test sets.
    Trained an XGBoostClassifier to predict 30-day readmission.
    Evaluated the model using accuracy, precision, recall, F1 score, and AUC-ROC.
    Saved feature importance plots and prediction functions.

2. Frontend (Data Analysis & Interactive Dashboard)
   
  a)Exploratory Data Analysis (Notebook)
  
    Class Balance Check: Visualized how many patients were readmitted within 30 days.
    Correlation Heatmaps: To identify relationships between features.    
    Bar Plots and Histograms: For age distribution, gender breakdown, and length of stay.
    Top Feature Visualizations: Feature importance plotted using sns.barplot.

   b)Interactive Data Exploration
   
     The dashboard offers rich interactivity powered by Plotly and Pandas:

     🔍 Overview Page
     
      Key metrics: average heart rate, blood pressure, oxygen level.
      Time-series trend charts for vitals.

      📊 Basic Analysis Tab
      
      Users can select:
      Custom date ranges
      Metrics like heart rate, BMI, etc.
      Time grouping (daily, weekly, monthly)
      Display summary statistics: mean, median, standard deviation.
      Distribution plots with adjustable bins.

      🧠 Advanced Insights Tab
      
      Correlation Matrix: Heatmap of inter-metric correlations.
      Risk Factor Analysis: Measures how individual features impact risk score.
      Anomaly Detection:
      Uses z-score logic (2 standard deviations) to find outliers.
      Visual markers highlight anomalies on time-series plots.

      🧑‍⚕️ Patient Comparison Tab
      
      Compares two patient groups (e.g., Male vs. Female) across:
      Heart rate
      Blood pressure
      Oxygen level
      BMI
      Risk score
      Radar chart and tabular comparisons.

     💡 Tools Used
     
      pandas / numpy: Data manipulation and numerical analysis.
      matplotlib / seaborn: Static plots and graphs in the notebook.
      plotly.express / plotly.graph_objects: Interactive plots in Streamlit.
      scikit-learn: For scaling, training, and evaluating models.
      streamlit: For building real-time, interactive dashboards.


🛠️STRUCTURE

CSV Files (MIMIC-III) 
    ↓
Data Cleaning & Feature Engineering (Notebook)
    ↓
Model Training (XGBoost)
    ↓
Model Evaluation (classification_report, AUC)
    ↓
Streamlit UI with RandomForest on Simulated Data
    ↓
Interactive Risk Prediction + Visualization + Health Tools


📈 Model Performance

Accuracy: ~86%
Precision: ~90%
f1 Score: ~87%
Confusion Matrix:  [[10  1]
                   [ 2  9]]
Key Features: Age, BMI, Heart Rate, Blood Pressure, Diabetes, Smoking, Cholesterol


🚀 Future Scope

As healthcare becomes increasingly data-driven, this project has strong potential to evolve into a real-world decision-support tool. Here are some ways this system can be extended and improved:

1. 🔬 Integration with Real-World Clinical Data
2. 🧠 Deep Learning Models
3. 📈 Time-Series Forecasting
4. 🧪 Explainable AI (XAI)
5. 📱 Mobile Health App
6. 🩺 Real-time Sensor Integration
7. 🏥 Clinician Dashboard
8. 📋 Automatic Emergency Alerts

