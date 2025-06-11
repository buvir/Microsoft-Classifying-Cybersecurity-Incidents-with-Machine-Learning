# %% [markdown]
# # Microsoft Cybersecurity Incident Classification
# ## Classifying incidents as TP, BP, or FP using Machine Learning

# %% [markdown]
# ## 1. Environment Setup and Data Download

# %%
import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# %%
# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# %%
def download_and_extract_zip(url, filename):
    """Download and extract zip file from Google Drive"""
    print(f"Downloading {filename}...")
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large file download
    zip_path = f"data/{filename}"
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
    
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    print(f"Completed {filename} processing")
    return f"data/{filename.replace('.zip', '.csv')}"

# %%
# Direct download links (replace with your actual links)
test_zip_url = 'https://drive.google.com/uc?export=download&id=1jtBLurdvjkBpzBhABJsRspNuN-mPs6Z5'
train_zip_url = 'https://drive.google.com/uc?export=download&id=1wKJgSOGjjKQh2CO3WaEi5L0S0ZuMnbUR'

# Download and extract files
test_csv_path = download_and_extract_zip(test_zip_url, 'test.zip')
train_csv_path = download_and_extract_zip(train_zip_url, 'train.zip')

# %%
# Load datasets
print("Loading datasets...")
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
# Initial inspection
print("Train dataset info:")
print(train_df.info())

# %%
# Target variable distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='triage_grade', data=train_df)
plt.title('Distribution of Triage Grades in Training Data')
plt.show()

# %%
# Check for missing values
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Missing values:\n", missing_values)

# %%
# Numerical features analysis
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
print(f"Numerical columns: {len(numerical_cols)}")

# Plot distributions for some numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols[:6]):
    plt.subplot(2, 3, i+1)
    sns.histplot(train_df[col], bins=30, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# %%
# Categorical features analysis
categorical_cols = train_df.select_dtypes(include=['object']).columns.drop('triage_grade')
print(f"Categorical columns: {len(categorical_cols)}")

# Plot value counts for some categorical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols[:6]):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=col, data=train_df, order=train_df[col].value_counts().iloc[:10].index)
    plt.xticks(rotation=45)
    plt.title(col)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Data Preprocessing

# %%
# Define features and target
X_train = train_df.drop(columns=['triage_grade'])
y_train = train_df['triage_grade']
X_test = test_df.drop(columns=['triage_grade'])
y_test = test_df['triage_grade']

# %%
# Identify feature types
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# %%
# Preprocessing pipeline
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %% [markdown]
# ## 4. Model Training

# %%
# Class weights calculation (to handle imbalance)
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# %%
# Baseline Random Forest model
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1))
])

print("Training Random Forest model...")
rf_model.fit(X_train, y_train)

# %%
# XGBoost model
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softmax',
        random_state=42,
        n_jobs=-1))
])

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# %% [markdown]
# ## 5. Model Evaluation

# %%
def evaluate_model(model, X, y, model_name):
    """Evaluate model and print metrics"""
    print(f"\nEvaluating {model_name} model...")
    y_pred = model.predict(X)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=classes))
    
    macro_f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    
    return {
        'model': model_name,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }

# %%
# Evaluate on validation set (20% of training data)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

rf_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest")
xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "XGBoost")

# %%
# Final evaluation on test set
print("\n\n=== Final Test Evaluation ===")
rf_test_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest (Test)")
xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Test)")

# %% [markdown]
# ## 6. Model Interpretation

# %%
# Feature importance for Random Forest
if hasattr(rf_model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    feature_names = (numerical_features.tolist() + 
                    list(rf_model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features)))
    
    importances = rf_model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.title('Random Forest - Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# %% [markdown]
# ## 7. Model Saving

# %%
# Save the best performing model
print("\nSaving the best model...")
joblib.dump(xgb_model, 'cybersecurity_incident_classifier.pkl')

# %%
# Save evaluation metrics
metrics_df = pd.DataFrame([rf_metrics, xgb_metrics, rf_test_metrics, xgb_test_metrics])
metrics_df.to_csv('model_evaluation_metrics.csv', index=False)
print("Evaluation metrics saved to model_evaluation_metrics.csv")

# %% [markdown]
# ## 8. Deployment Recommendations

# %%
print("""
## Deployment Recommendations:

1. **Model Integration**: 
   - Integrate the trained model into SOC workflows as a triage assistant
   - Use model predictions to prioritize incident review queues

2. **Monitoring**:
   - Implement continuous performance monitoring
   - Set up alerts for model drift or performance degradation

3. **Feedback Loop**:
   - Collect analyst feedback on predictions
   - Use this to periodically retrain the model

4. **Scalability**:
   - For production, consider containerizing the model
   - Use a microservices architecture for scalability

5. **Security**:
   - Ensure model API has proper authentication
   - Log all prediction requests for audit purposes
""")

# %% [markdown]
# ## 9. Project Documentation

print("""
# Project Documentation: Microsoft Cybersecurity Incident Classification

## Overview
This project developed a machine learning model to classify cybersecurity incidents as:
- True Positive (TP)
- Benign Positive (BP)
- False Positive (FP)

## Key Steps
1. Data acquisition and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model training (Random Forest and XGBoost)
5. Model evaluation
6. Model interpretation

## Results
The best performing model achieved:
- Macro F1 Score: {xgb_test_metrics['macro_f1']:.4f}
- Precision: {xgb_test_metrics['precision']:.4f}
- Recall: {xgb_test_metrics['recall']:.4f}

## Files Included
1. `cybersecurity_incident_classifier.pkl` - Trained model
2. `model_evaluation_metrics.csv` - Performance metrics
3. This notebook - Complete project documentation

## Future Improvements
1. Experiment with deep learning models
2. Incorporate more feature engineering
3. Implement automated retraining pipeline
""")
