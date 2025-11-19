
# Bank Transaction Fraud Detection

An end-to-end ML project for detecting fraudulent transactions using machine learning.
Built with Python, FastAPI, and Docker.

## Problem Description
Financial institutions lose billions yearly to fraudulent transactions. __[bank transaction dataset](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection?resource=download)__
The goal of this project is to develop a machine learning model that identifies potentially fraudulent transactions using historical behavioral data.

The system is designed to:

- Load and preprocess raw transaction data  
- Perform cleaning, handling missing values, and scaling numerical features  
- Engineer additional predictive features based on user behavior and transaction patterns  
- Train multiple machine learning models and compare their performance  
- Automatically select and export the best-performing model  
- Provide a fully functional **FastAPI web service** for real-time predictions  
- Offer an optional Docker-based deployment workflow for portability

This project demonstrates strong understanding of data processing, feature engineering, model training, backend model serving, and deployment fundamentals. The repository is structured to follow best-practice ML engineering principles, making it easy to extend, reproduce, and deploy.


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
Follow the steps below to set up and run the full pipeline on your local machine.
1. Create virtual environment  
2. Install dependencies: `pip install -r requirements.txt` All project dependencies are listed in requirements.txt.
3. Run training: `python src/model_trainer.py` This script loads data, performs feature engineering, trains multiple models, selects the best one, and saves it as best_model.pkl.
4. Launch API: `uvicorn app.main:app --reload`
5. Test with Postman or `curl`

##  References
DataTalksClub (2025), Machine Learning Zoomcamp Course Materials. Available at: https://github.com/DataTalksClub/machine-learning-zoomcamp.
