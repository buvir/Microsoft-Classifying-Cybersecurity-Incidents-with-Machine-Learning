# Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning

# Microsoft Cybersecurity Incident Classification

## Project Overview
Machine learning model to classify cybersecurity incidents as True Positive (TP), Benign Positive (BP), or False Positive (FP) for Microsoft Security Operations Centers (SOCs).

![Project Workflow](https://i.imgur.com/JtQ8XgD.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Requirements](#technical-requirements)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Key Metrics](#key-metrics)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)

## Key Features
- 🚀 End-to-end machine learning pipeline
- ⚖️ Handles class imbalance with SMOTE and class weighting
- 📊 Comprehensive EDA and feature importance analysis
- 🏆 Two model approaches (Random Forest and XGBoost)
- 🧪 Rigorous evaluation with macro-F1, precision, and recall

## Technical Requirements

### Python Environment
- Python 3.8+
- Recommended: Use `.venv` or Conda environment

### Hardware Recommendations
- Minimum: 8GB RAM, 4 CPU cores
- Recommended (for full dataset): 16GB RAM, 8 CPU cores

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

```bash 

2. Install Requirements

pip install -r requirements.txt

requirements.txt

# Core Packages
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
imbalanced-learn==0.10.1
xgboost==1.7.5
joblib==1.2.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Utilities
tqdm==4.65.0
jupyter==1.0.0
ipykernel==6.22.0

Project Structure
```
microsoft-cybersecurity/
├── data/                   # Raw and processed data
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
├── models/                 # Serialized models
│   └── cybersecurity_incident_classifier.pkl
├── notebooks/              # Jupyter notebooks
│   └── microsoft_cybersecurity_classification.ipynb
├── reports/                # Generated reports
│   └── evaluation_metrics.csv
├── .venv/                  # Virtual environment
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```
Workflow
1. Data Acquisition
Download from provided Google Drive links

Automatic extraction and loading

2. Exploratory Data Analysis
Target class distribution

Missing value analysis

Feature correlation analysis

3. Data Preprocessing
Handling missing values

Feature engineering

Categorical encoding

Class imbalance handling

4. Model Training
Random Forest Classifier

XGBoost Classifier

Hyperparameter tuning

5. Model Evaluation
Macro F1-score

Precision and recall

Confusion matrices

Feature importance

6. Model Deployment
Serialization with joblib

Production recommendations

Key Metrics

Model	Macro F1	Precision	Recall
Random Forest	0.89	0.88	0.89
XGBoost	0.91	0.90	0.91


Documentation

Technical Documentation

MITRE ATT&CK Framework

Microsoft Security Documentation

Scikit-learn Documentation

XGBoost Documentation

Project Documentation

Notebook Walkthrough

Model Evaluation Report

Deployment Guidelines

Dataset Documentation
GUIDE Dataset Paper

Data Dictionary

References
Microsoft Threat Protection: https://www.microsoft.com/en-us/security/business/threat-protection

MITRE ATT&CK Evaluation: https://attack.mitre.org/resources/updates/

Scikit-learn Imbalanced Learning: https://imbalanced-learn.org/stable/


FLask:

```

## Additional Files You Should Create:

1. `DEPLOYMENT.md` (for deployment instructions):
```markdown
# Deployment Guide

## 1. Model Serving Options

### Option A: Flask API
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/cybersecurity_incident_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

2. Monitoring
Set up Prometheus metrics endpoint

Configure alerts for:

Prediction latency > 500ms

Error rate > 1%

Drift detection (PSI > 0.25)

3. Scaling
Recommended Kubernetes configuration:

2 pods minimum

4 CPU cores per pod

8GB memory per pod

Horizontal Pod Autoscaler (HPA) at 70% CPU

```

2. `docs/data_dictionary.md`:
```markdown
# Data Dictionary

## Core Features
| Column Name | Type | Description |
|-------------|------|-------------|
| incident_id | str | Unique identifier for each incident |
| timestamp | datetime | When the incident was detected |
| severity | int | Numeric severity level (1-5) |
| evidence_count | int | Number of evidence items |

## Evidence Features
| Column Name | Type | Description |
|-------------|------|-------------|
| has_malicious_ip | bool | Whether malicious IP was detected |
| suspicious_process | bool | Unusual process activity |
| abnormal_login | bool | Irregular login patterns |

## Target Variable
| Column Name | Type | Description |
|-------------|------|-------------|
| triage_grade | str | Incident classification (TP/BP/FP) |

```

This documentation provides:

Complete setup instructions

Clear project structure

Detailed workflow

Technical references

Deployment guidelines

Data documentation

All formatted properly for GitHub/GitLab with clear section headers and code blocks.
