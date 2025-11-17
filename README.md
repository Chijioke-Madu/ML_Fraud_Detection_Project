
# Bank Transaction Fraud Detection

An end-to-end ML project for detecting fraudulent transactions using machine learning.
Built with Python, FastAPI, and Docker.

## Problem Description
Financial institutions lose billions yearly to fraudulent transactions. __[bank transaction dataset] (https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection?resource=download)__
The goal of this project is to develop a machine learning model that identifies potentially fraudulent transactions using historical behavioral data.

## Features
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- REST API for real-time predictions
- Docker containerization

## Notebook: Data Preparation & EDA
Open the notebook:
jupyter notebook notebook.ipynb
The notebook includes:
Data cleaning
Exploratory data analysis
Feature engineering
Model comparisons
Metrics (Accuracy, Precision, Recall, F1-score)

## How to Run
1. Create virtual environment  
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/model_trainer.py`
4. Launch API: `uvicorn app.main:app --reload`
5. Test with Postman or `curl`
